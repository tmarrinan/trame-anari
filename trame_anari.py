import sys
sys.path.append('/home/tmarrinan/local/lib')

import asyncio
import math
import time
import io
from trame.app import get_server, asynchronous
from trame.widgets import vuetify, rca, client
from trame.ui.vuetify import SinglePageLayout
import numpy as np
import pynari as anari
from PIL import Image
from mpi4py import MPI


def main():
    # initialize MPI
    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    mpi_size = comm.Get_size()

    # create view for custom ANARI application
    view = AnariView(mpi_rank, mpi_size, comm)

    # main task initializes Trame server
    if mpi_rank == 0:
        setupTrameServer(view)
    # other tasks wait for signal to rerender or quit
    else:
        finished = False
        while not finished:
            signal = np.empty(3, dtype=np.int16)
            comm.Bcast((signal, 3, MPI.INT16_T), root=0)
            if signal[0] == 0:    # quit
                finished = True
            elif signal[0] == 1:  # rerender
                view.render()
            elif signal[0] == 2:  # resize
                view.setRenderSize(int(signal[1]), int(signal[2]))
            elif signal[0] == 3:  # rotate camera
                view.rotateCamera(int(signal[1]), int(signal[2]))

def setupTrameServer(view):
    # set up Trame application
    server = get_server(client_type='vue2')
    state = server.state
    ctrl = server.controller

    # register RCA view with Trame controller
    view_handler = None
    @ctrl.add('on_server_ready')
    def initRca(**kwargs):
        nonlocal view_handler
        view_handler = RcaViewAdapter(view, 'view')
        ctrl.rc_area_register(view_handler)

    # callback for change in number of path tracing samples per pixel
    def uiStateNumSamplesUpdate(num_samples, **kwargs):
        view.setNumberOfSamples(num_samples)
        if view_handler is not None:
            view_handler.pushFrame()

    #register callbacks
    state.change('num_samples')(uiStateNumSamplesUpdate)

    # define webpage layout
    with SinglePageLayout(server) as layout:
        layout.title.set_text('Trame-ANARI')
        with layout.toolbar:
            vuetify.VDivider(vertical=True, classes='mx-2')
            vuetify.VSlider(
                label='Number of Samples',
                v_model=('num_samples', 4),
                min=1,
                max=32,
                step=1,
                hide_details=True,
                dense=True
            )
            vuetify.VCol(
                '{{num_samples}}'
            )
        with layout.content:
            with vuetify.VContainer(fluid=True, classes='pa-0 fill-height'):
                v = rca.RemoteControlledArea(name='view', display='image', id='rca-view')

    # start Trame server
    server.start()


# Trame RCA View Adapter
class RcaViewAdapter:
    def __init__(self, view, name):
        self._view = view
        self._streamer = None
        self._metadata = {
            'type': 'image/jpeg',
            'codec': '',
            'w': 0,
            'h': 0,
            'st': 0,
            'key': 'key'
        }

        self.area_name = name

    def pushFrame(self):
        if self._streamer is not None:
            asynchronous.create_task(self._asyncPushFrame())

    async def _asyncPushFrame(self):
        frame_data = self._view.getFrame()
        self._streamer.push_content(self.area_name, self._getMetadata(), frame_data.data)

    def _getMetadata(self):
        width, height = self._view.getSize()
        self._metadata['w'] = width
        self._metadata['h'] = height
        self._metadata['st'] = self._view.getFrameTime()
        return self._metadata

    def set_streamer(self, stream_manager):
        self._streamer = stream_manager

    def update_size(self, origin, size):
        width = int(size.get('w', 400))
        height = int(size.get('h', 300))
        self._view.triggerResize(width, height)
        self._view.triggerRender()
        self.pushFrame()
        print(f'new size: {width}x{height}')
        sys.stdout.flush()

    def on_interaction(self, origin, event):
        event_type = event['type']
        rerender = False

        if event_type == 'LeftButtonPress':
            rerender = self._view.onLeftMouseButton(event['x'], event['y'], True)
        elif event_type == 'LeftButtonRelease':
            rerender = self._view.onLeftMouseButton(event['x'], event['y'], False)
        elif event_type == 'MouseMove':
            rerender = self._view.onMouseMove(event['x'], event['y'])

        if rerender:
            self._view.triggerRender()
            frame_data = self._view.getFrame()
            self._streamer.push_content(self.area_name, self._getMetadata(), frame_data.data)


# Trame custom ANARI view
class AnariView:
    def __init__(self, mpi_rank, mpi_size, comm):
        self._task_id = mpi_rank
        self._num_tasks = mpi_size
        self._mpi_comm = comm

        # user interaction
        self._rotate_camera = False
        self._mouse_pos = (0, 0)

        # store time frame is rendered at
        self._frame_time = round(time.time_ns() / 1000000)
        
        # create ANARI device
        self._device = anari.newDevice('default')

        # initial framebuffer size
        self._framebuffer_size = (512, 512)
        
        # initial camera parameters
        self._cam_theta = math.radians(-15.0)
        self._cam_phi = math.radians(90.0)
        self._cam_radius = 9.0
        #self._cam_position = (-2.5, 3.5, 7.5)
        self._cam_target = (0.0, 0.0, 0.0)
        self._cam_up = (0.0, 1.0, 0.0)
        self._fovy = math.radians(40.0)
        cam_position = self._calculateCameraPosition()

        # initial number of ray samples per pixel
        self._ray_samples = 4

        # add geometry to scene
        surfaces = self._createSurfaces()

        self._world = self._device.newWorld()
        self._world.setParameterArray('surface', anari.SURFACE, surfaces)
        self._world.commitParameters()

        # set up camera
        self._camera = self._device.newCamera('perspective')
        self._camera.setParameter('aspect', anari.FLOAT32, self._framebuffer_size[0] / self._framebuffer_size[1])
        self._camera.setParameter('position',anari.FLOAT32_VEC3, cam_position)
        direction = [self._cam_target[0] - cam_position[0],
                     self._cam_target[1] - cam_position[1],
                     self._cam_target[2] - cam_position[2]]
        self._camera.setParameter('direction', anari.float3, direction)
        self._camera.setParameter('up', anari.float3, self._cam_up)
        self._camera.setParameter('fovy', anari.FLOAT32, self._fovy)
        self._camera.commitParameters()

        # background gradient - light gray to blue (image 1 px wide, 2 px tall)
        bg_values = np.array(((0.9, 0.9, 0.9, 1.0), (0.15, 0.25, 0.8, 1.0)), dtype=np.float32).reshape((4, 1, 2))
        bg_gradient = self._device.newArray(anari.float4, bg_values)

        # create renderer and set background
        self._renderer = self._device.newRenderer('default')
        self._renderer.setParameter('ambientRadiance', anari.FLOAT32, 1.0)
        self._renderer.setParameter('background', anari.ARRAY, bg_gradient)
        self._renderer.setParameter('pixelSamples', anari.INT32, self._ray_samples)
        self._renderer.commitParameters()

        # create frame
        self._frame = self._device.newFrame()
        self._frame.setParameter('size', anari.uint2, self._framebuffer_size)
        self._frame.setParameter('channel.color', anari.DATA_TYPE, anari.UFIXED8_VEC4)
        self._frame.setParameter('renderer', anari.OBJECT, self._renderer)
        self._frame.setParameter('camera', anari.OBJECT, self._camera)
        self._frame.setParameter('world', anari.OBJECT, self._world)
        self._frame.commitParameters()

        # render image
        self.render()

    #
    def triggerRender(self):
        self._mpi_comm.Bcast((np.array([1, 0, 0], dtype=np.int16), 3, MPI.INT16_T), root=0)
        self.render()

    #
    def triggerResize(self, width, height):
        self._mpi_comm.Bcast((np.array([2, width, height], dtype=np.int16), 3, MPI.INT16_T), root=0)
        self.setRenderSize(width, height)

    # render frame
    def render(self):
        self._frame_time = round(time.time_ns() / 1000000)
        self._frame.render()

    # get width and height of frame
    def getSize(self):
        return self._framebuffer_size

    # get time frame was rendered at
    def getFrameTime(self):
        return self._frame_time

    # get frame encoded as JPEG
    def getFrame(self):
        if self._task_id == 0:
            anari_composite_image = io.BytesIO()
            pixels = np.array(self._frame.get('channel.color'))
            img = Image.fromarray(pixels)
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            img = img.convert('RGB')
            img.save(anari_composite_image, 'JPEG')
            return np.frombuffer(anari_composite_image.getbuffer(), dtype=np.uint8)
        else:
            return None

    # set number of samples
    def setNumberOfSamples(self, num):
        self._ray_samples = num

    # set render frame size
    def setRenderSize(self, width, height):
        self._framebuffer_size = (width, height)
        self._camera.setParameter('aspect', anari.FLOAT32, self._framebuffer_size[0] / self._framebuffer_size[1])
        self._camera.commitParameters()
        self._frame.setParameter('size', anari.uint2, self._framebuffer_size)
        self._frame.commitParameters()

    # handler for left mouse button -> return whether or not rerender is required
    def onLeftMouseButton(self, mouse_x, mouse_y, pressed):
        if pressed:
            self._rotate_camera = True
            self._mouse_pos = (mouse_x, mouse_y)
        else:
            self._rotate_camera = False
        return False

    # handler for mouse movement -> return whether or not rerender is required
    def onMouseMove(self, mouse_x, mouse_y):
        if self._rotate_camera:
            delta_x = mouse_x - self._mouse_pos[0]
            delta_y = mouse_y - self._mouse_pos[1]
            self._mpi_comm.Bcast((np.array([3, delta_x, delta_y], dtype=np.int16), 3, MPI.INT16_T), root=0)
            self.rotateCamera(delta_x, delta_y)
            return True
        else:
            return False

    # rotate camera
    def rotateCamera(self, delta_x, delta_y):
        self._cam_theta -= math.radians(delta_x * 0.01)
        self._cam_phi = min(max(self._cam_phi + math.radians(delta_y * 0.01), math.radians(1.0)), math.radians(179.0))
        cam_position = self._calculateCameraPosition()
        self._camera.setParameter('position',anari.FLOAT32_VEC3, cam_position)
        direction = [self._cam_target[0] - cam_position[0],
                     self._cam_target[1] - cam_position[1],
                     self._cam_target[2] - cam_position[2]]
        self._camera.setParameter('direction', anari.float3, direction)
        self._camera.commitParameters()

    # calculate camera position based on spherical coords
    def _calculateCameraPosition(self):
        x = self._cam_radius * math.sin(self._cam_phi) * math.sin(self._cam_theta)
        y = self._cam_radius * math.cos(self._cam_phi)
        z = self._cam_radius * math.sin(self._cam_phi) * math.cos(self._cam_theta)
        return (x, y, z)

    # create ANARI surfaces
    def _createSurfaces(self):
        total_cubes = 8
        cube_centers = [
            (-0.8, -0.8, -0.8),
            ( 0.8, -0.8, -0.8),
            (-0.8,  0.8, -0.8),
            ( 0.8,  0.8, -0.8),
            (-0.8, -0.8,  0.8),
            ( 0.8, -0.8,  0.8),
            (-0.8,  0.8,  0.8),
            ( 0.8,  0.8,  0.8)
        ]

        surfaces = []

        geom = self._device.newGeometry('triangle')
        vertices = []
        indices = []

        num_cubes = total_cubes // self._num_tasks
        start_idx = self._task_id * num_cubes
        for i in range(num_cubes):
            center = cube_centers[start_idx + i]
            vertices.append((-0.5 + center[0], -0.5 + center[1], -0.5 + center[2]))
            vertices.append(( 0.5 + center[0], -0.5 + center[1], -0.5 + center[2]))
            vertices.append((-0.5 + center[0],  0.5 + center[1], -0.5 + center[2]))
            vertices.append(( 0.5 + center[0],  0.5 + center[1], -0.5 + center[2]))
            vertices.append((-0.5 + center[0], -0.5 + center[1],  0.5 + center[2]))
            vertices.append(( 0.5 + center[0], -0.5 + center[1],  0.5 + center[2]))
            vertices.append((-0.5 + center[0],  0.5 + center[1],  0.5 + center[2]))
            vertices.append(( 0.5 + center[0],  0.5 + center[1],  0.5 + center[2]))
            indices.append((8 * i + 1, 8 * i + 0, 8 * i + 3))
            indices.append((8 * i + 0, 8 * i + 2, 8 * i + 3))
            indices.append((8 * i + 5, 8 * i + 1, 8 * i + 7))
            indices.append((8 * i + 1, 8 * i + 3, 8 * i + 7))
            indices.append((8 * i + 0, 8 * i + 4, 8 * i + 2))
            indices.append((8 * i + 4, 8 * i + 6, 8 * i + 2))
            indices.append((8 * i + 4, 8 * i + 5, 8 * i + 6))
            indices.append((8 * i + 5, 8 * i + 7, 8 * i + 6))
            indices.append((8 * i + 0, 8 * i + 1, 8 * i + 4))
            indices.append((8 * i + 1, 8 * i + 5, 8 * i + 4))
            indices.append((8 * i + 6, 8 * i + 7, 8 * i + 2))
            indices.append((8 * i + 7, 8 * i + 3, 8 * i + 2))

        mesh_vertices = self._device.newArray(anari.FLOAT32_VEC3, np.array(vertices, dtype=np.float32).flatten())
        mesh_indices = self._device.newArray(anari.UINT32_VEC3, np.array(indices, dtype=np.uint32).flatten())

        geom.setParameter('vertex.position', anari.ARRAY, mesh_vertices)
        geom.setParameter('primitive.index', anari.ARRAY, mesh_indices)
        geom.commitParameters()

        material = self._makeMaterial()
        surf = self._device.newSurface()
        surf.setParameter('geometry', anari.GEOMETRY, geom)
        surf.setParameter('material', anari.MATERIAL, material)
        surf.commitParameters()

        surfaces.append(surf)

        return surfaces

    # create material
    def _makeMaterial(self):
        task_colors = [
            ( 25,  25, 112), (  4, 100,   9), (186,  35,  35), (240, 214,   9),
            (118, 214,  80), ( 42, 206, 206), (179,  61, 184), (255, 182, 177)
        ]
        rgb = (
            (task_colors[self._task_id][0] / 255) ** 2.2,
            (task_colors[self._task_id][1] / 255) ** 2.2,
            (task_colors[self._task_id][2] / 255) ** 2.2
        )

        mat = self._device.newMaterial('physicallyBased')
        mat.setParameter('baseColor', anari.float3, rgb)
        mat.setParameter('ior', anari.FLOAT32, 1.45)
        mat.setParameter('metallic', anari.FLOAT32, 0.0)
        mat.setParameter('specular', anari.FLOAT32, 0.0)
        mat.setParameter('roughness', anari.FLOAT32, 0.8)
        mat.commitParameters()

        return mat


# run `main()` if primary script
if __name__ == '__main__':
    main()

