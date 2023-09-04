import numpy as np
import trimesh

# attach to logger so trimesh messages will be printed to console
trimesh.util.attach_to_log()

# mesh objects can be created from existing faces and vertex data
mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [0, 0, 1], [0, 1, 0]],
                       faces=[[0, 1, 2]])

# by default, Trimesh will do a light processing, which will
# remove any NaN values and merge vertices that share position
# if you want to not do this on load, you can pass `process=False`
mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [0, 0, 1], [0, 1, 0]],
                       faces=[[0, 1, 2]],
                       process=False)

# some formats represent multiple meshes with multiple instances
# the loader tries to return the datatype which makes the most sense
# which will for scene-like files will return a `trimesh.Scene` object.
# if you *always* want a straight `trimesh.Trimesh` you can ask the
# loader to "force" the result into a mesh through concatenation
mesh = trimesh.load('butterfly.glb', force='mesh')

# is the current mesh watertight?
mesh.is_watertight

# what's the euler number for the mesh?
mesh.euler_number

# the convex hull is another Trimesh object that is available as a property
# lets compare the volume of our mesh with the volume of its convex hull
print(mesh.volume / mesh.convex_hull.volume)

# since the mesh is watertight, it means there is a
# volumetric center of mass which we can set as the origin for our mesh
mesh.vertices -= mesh.center_mass

# what's the moment of inertia for the mesh?
mesh.moment_inertia

# if there are multiple bodies in the mesh we can split the mesh by
# connected components of face adjacency
# since this example mesh is a single watertight body we get a list of one mesh
mesh.split()

# facets are groups of coplanar adjacent faces
# set each facet to a random color
# colors are 8 bit RGBA by default (n, 4) np.uint8
for facet in mesh.facets:
    mesh.visual.face_colors[facet] = trimesh.visual.random_color()

# preview mesh in an opengl window if you installed pyglet and scipy with pip
mesh.show()

# transform method can be passed a (4, 4) matrix and will cleanly apply the transform
mesh.apply_transform(trimesh.transformations.random_rotation_matrix())

# axis aligned bounding box is available
mesh.bounding_box.extents

# a minimum volume oriented bounding box also available
# primitives are subclasses of Trimesh objects which automatically generate
# faces and vertices from data stored in the 'primitive' attribute
mesh.bounding_box_oriented.primitive.extents
mesh.bounding_box_oriented.primitive.transform

# show the mesh appended with its oriented bounding box
# the bounding box is a trimesh.primitives.Box object, which subclasses
# Trimesh and lazily evaluates to fill in vertices and faces when requested
# (press w in viewer to see triangles)
(mesh + mesh.bounding_box_oriented).show()

# bounding spheres and bounding cylinders of meshes are also
# available, and will be the minimum volume version of each
# except in certain degenerate cases, where they will be no worse
# than a least squares fit version of the primitive.
print(mesh.bounding_box_oriented.volume,
      mesh.bounding_cylinder.volume,
      mesh.bounding_sphere.volume)


"""
import numpy as np
import trimesh
import pyvista as pv
import pyvistaqt as pvqt
mesh = trimesh.load('butterfly.glb', force='mesh')
# 这里是给每个顶点赋值，所以他的颜色是渐变的，如果给cell赋值，那么每个格子的颜色就是一样的。
mesh.point_arrays['my point values'] = np.arange(mesh.n_points)
mesh.plot(scalars='my point values', show_edges=True, screenshot='beam_point_data.png')
"""
"""
######################################################
pl = pv.Plotter() # Plotter class 实例化
pl.add_mesh(mesh, show_edges=True)
pl.show()
#######################################################
plotter = pvqt.BackgroundPlotter()
plotter.add_mesh(mesh, lighting=False, show_edges=True, texture=True, scalars='scalars')
plotter.view_isometric()
plotter.show()
######################################################
boundary = mesh.bounds
print(boundary)
v = mesh.vertices
f = mesh.faces

v = np.array(v)
f = np.array(f)
print('v:',v)
print(v.shape)
print('f:',f)
print(f.shape)
mesh = trimesh.Trimesh(vertices=v, faces=f)
mesh.show()
######################################################################
import trimesh
import numpy as np

v = [[1, 0, 0], [1, 1, 0], [0, 1, 0], [1, 1, 1]]
f = [[0, 1, 3], [0, 1, 3], [1, 2, 3], [0, 2, 3]]
mesh = trimesh.Trimesh(vertices=v, faces=f)

# by default, Trimesh will do a light processing, which will
# remove any NaN values and merge vertices that share position
# if you want to not do this on load, you can pass `process=False`

mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
mesh.show()
"""