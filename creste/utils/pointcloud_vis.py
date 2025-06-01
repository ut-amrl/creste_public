"""
Borrowed from TerrainNet Implementation
"""
import numba
import numpy as np
from matplotlib import pyplot as plt

import vispy
from vispy.color.color_array import Color
from vispy.geometry import create_sphere, create_arrow
from vispy.geometry.meshdata import MeshData
from vispy.scene import visuals, SceneCanvas
from vispy.visuals.filters import ShadingFilter
from vispy.visuals.transforms import MatrixTransform


@numba.jit(nopython=True)
def _draw_mesh_grid_helper(grid_points, mask, resolution, colors):
    # Here we try to avoid creating numpy arrays in the loops
    # so most of the computation is done explicitly.
    # This provides a 3X speedup.
    def _compute_normal(p1, p2, p3):
        a = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
        b = (p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2])
        # cross product
        n = (a[1] * b[2] - a[2] * b[1],
             a[2] * b[0] - a[0] * b[2],
             a[0] * b[1] - a[1] * b[0])
        norm = np.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
        if norm < 1e-5:
            return 0, 0, 0
        return n[0] / norm, n[1] / norm, n[2] / norm

    h, w = grid_points.shape[:2]
    vertices = []
    # Need to hint the type to circumvent compilation errors
    vertex_colors = [(np.float32(x), np.float32(x), np.float32(x)) for x in range(0)]
    faces = []
    normals = []
    s = 0
    for i in range(h - 1):
        for j in range(w - 1):
            if not (mask[i, j] and mask[i + 1, j]
                    and mask[i + 1, j + 1] and mask[i, j + 1]):
                continue
            y1 = (i - h / 2) * resolution
            x1 = (j - w / 2) * resolution
            z1 = grid_points[i, j]

            y2 = (i + 1 - h / 2) * resolution
            x2 = (j - w / 2) * resolution
            z2 = grid_points[i + 1, j]

            y3 = (i + 1 - h / 2) * resolution
            x3 = (j + 1 - w / 2) * resolution
            z3 = grid_points[i + 1, j + 1]

            y4 = (i - h / 2) * resolution
            x4 = (j + 1 - w / 2) * resolution
            z4 = grid_points[i, j + 1]

            vertices.extend([(x4, y4, z4), (x2, y2, z2), (x1, y1, z1),
                             (x4, y4, z4), (x3, y3, z3), (x2, y2, z2)])
            faces.extend([(s, s + 1, s + 2), (s + 3, s + 4, s + 5)])

            if colors is not None:
                # The order of the offsets must be consistent with the order
                # of the vertices
                for offset in ((0, 1), (1, 0), (0, 0), (0, 1), (1, 1), (1, 0)):
                    rgb = colors[i + offset[0], j + offset[1]]
                    vertex_colors.append((rgb[0], rgb[1], rgb[2]))

            # Vertex order matters, otherwise the rendering would look bad.
            n1 = _compute_normal((x4, y4, z4), (x2, y2, z2), (x1, y1, z1))
            n2 = _compute_normal((x4, y4, z4), (x3, y3, z3), (x2, y2, z2))
            normals.extend([n1, n1, n1, n2, n2, n2])

            s += 6

    vertices = np.array(vertices, np.float32)
    faces = np.array(faces, np.int32)
    normals = np.array(normals, np.float32)
    if len(vertex_colors) > 0:
        vertex_colors = np.array(vertex_colors, np.float32)
    else:
        vertex_colors = None
    return vertices, faces, normals, vertex_colors


class MyMeshData(MeshData):
    """
    Accept externally computed vertex normals.
    This would skip the internal vertex normal computation which
    is very slow. This provides a 100x speedup.
    """
    def __init__(self, *args, vertex_normals=None, **kwargs):
        super(MyMeshData, self).__init__(*args, **kwargs)
        if vertex_normals is not None:
            self._vertex_normals = vertex_normals

class LaserScanVis(object):
    def __init__(self, width=800, height=600, interactive=False, bg_color='k'):
        self.interactive = interactive
        self.cmap = self.get_mpl_colormap('rainbow')
        self.width = width
        self.height = height
        self.mesh_data = None
        self.mesh = None
        self._make_canvas(width, height, bg_color)

    def _make_canvas(self, width, height, bg_color):
        if self.interactive:
            kwargs = {
                'keys': 'interactive',
                'show': True
            }
        else:
            kwargs = {
                'show': False
            }
        self.canvas = SceneCanvas(size=(width, height), bgcolor=bg_color, **kwargs)

        self.canvas.events.key_press.connect(self.key_press)
        self.grid = self.canvas.central_widget.add_grid()

        # laserscan part
        self.scan_view = vispy.scene.widgets.ViewBox(
            border_width=0, parent=self.canvas.scene
        )
        self.grid.add_widget(self.scan_view, 0, 0)
        # Turning on antialiasing looks weird when doing off-screen rendering.
        # We'd rather turn it off
        self.scan_vis = visuals.Markers(antialias=0, alpha=1.0)
        self.scan_view.camera = 'turntable'
        self.scan_view.add(self.scan_vis)

        # Visualize the origin
        # mdata = create_sphere(20, 40, 1.0, radius=1.0)
        # self.origin_sphere = self.create_mesh(mdata, shading='flat')
        # self.origin_sphere.color = Color([1.0, 0.0, 0.0])
        # self.origin_sphere.set_gl_state('translucent')
        # self.scan_view.add(self.origin_sphere)

        # The mesh object doesn't accept an empty mesh,
        # so here we initially create a dummy sphere mesh.
        # Later we replace it with the actual mesh.
        mdata = create_sphere(20, 40, 1.0, radius=0.1)
        self.mesh = self.create_mesh(mdata)
        self.scan_view.add(self.mesh)

        # Arrow for visualizing vehicle's pose
        mdata = create_arrow(10, 10, 1.0, 20.0)
        self.arrow = self.create_mesh(mdata, shading='flat')
        self.arrow.color = Color([1.0, 0.0, 0.0])
        self.scan_view.add(self.arrow)
        self.arrow.visible = False

        self.axis = visuals.XYZAxis(parent=self.scan_view.scene)
        t = MatrixTransform()
        t.scale([5, 5, 5])
        t.translate([0, 0, 0])
        self.axis.transform = t

    def create_mesh(self, mesh_data, shading='smooth'):
        mesh = visuals.Mesh(meshdata=mesh_data)
        mesh.clim = [-1, 1]
        mesh.cmap = 'rainbow'
        smooth_shading = ShadingFilter(shading=shading, shininess=1000, ambient_light=(1, 1, 1, 0.7))
        mesh.attach(smooth_shading)
        return mesh

    def set_camera(self, camera_param):
        self.scan_view.camera.set_state(camera_param)
        # self.canvas.update()
        # self.scan_view.camera.view_changed()

    def get_mpl_colormap(self, cmap_name):
        cmap = plt.get_cmap(cmap_name)
        # Initialize the matplotlib color map
        sm = plt.cm.ScalarMappable(cmap=cmap)
        # Obtain linear color range
        color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]
        return color_range.reshape(256, 3).astype(np.float32) / 255.0

    def draw_points(self, points, colors=None, size=1, color_z_range=(-50, 50)):
        if colors is None and len(points) > 0:
            zs = points[:, 2]
            z_min = color_z_range[0]
            z_max = color_z_range[1]
            normalized_range = (np.clip(zs, z_min, z_max) - z_min) / (z_max - z_min)
            normalized_range = (normalized_range * 255).astype(np.uint8)
            colors = self.cmap[normalized_range][..., ::-1]

        self.scan_vis.set_data(
            points[:, :3],
            face_color=colors,
            edge_width=0,
            size=size)

    def draw_mesh_grid(self, grid_points, mask, resolution, colors=None):
        """
        Args:
            grid_points: a HxW array of z values. x and y values are calculated based on resolution.
            mask: a HxW boolean array
            resolution: resolution of each cell
            colors: (optional) a HxWx3 float32 array of RGB colors

        Returns: None

        """
        assert len(grid_points.shape) == 2, 'grid_points.shape != 2'
        if colors is not None:
            assert colors.shape[:2] == grid_points.shape[:2], \
                'colors shape must match grid_points shape in the first 2 dims'
        vertices, faces, normals, vertex_colors = _draw_mesh_grid_helper(
            grid_points, mask, resolution, colors)
        normalized_z = vertices[:, 2] / 20.0
        self.mesh_data = MyMeshData(vertices=vertices, faces=faces, vertex_colors=vertex_colors,
                                    vertex_values=normalized_z, vertex_normals=normals)
        self.mesh.set_data(meshdata=self.mesh_data)

    def set_arrow_pose(self, pose, z_adj=10.0):
        arrow_pose = np.eye(4)
        # The default arrow points upward. Need to rotate it to make it point
        # towards the x-axis
        rotate_y_90deg = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], np.float32)
        arrow_pose[:3, :3] = pose[:3, :3] @ rotate_y_90deg
        arrow_pose[2, 3] = z_adj
        # Need to transpose due to OpenGL matrix order convention
        t = MatrixTransform(arrow_pose.T)
        self.arrow.transform = t

    def set_points_visible(self, visible):
        self.scan_vis.visible = visible

    def set_mesh_visible(self, visible):
        self.mesh.visible = visible
        # self.origin_sphere.visible = visible

    def set_arrow_visible(self, visible):
        self.arrow.visible = visible

    def render(self):
        return self.canvas.render()

    def key_press(self, event):
        print('camera parameter:', self.scan_view.camera.get_state())
        self.canvas.events.key_press.block()
        if event.key == "Q" or event.key == "Escape":
            self.destroy()

    def destroy(self):
        # destroy the visualization
        self.canvas.close()
        self.canvas = None
        vispy.app.quit()

    def show(self, pause=True):
        if pause:
            vispy.app.run()
        else:
            vispy.app.process_events()
