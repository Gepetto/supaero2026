"""
Simple example of use of display_collision_patches.
Create a scene with 3 objects, move them and display the collision patches during the
movement.

import time

import numpy as np
import pinocchio as pin
from create_rigid_contact_models_for_hppfcl import (
    createContactModelsFromCollisions,
    createContactModelsFromDistances,
)
from display_collision_patches import (
    COLPATCH_DEFAULT_PREALLOC,
    preallocateVisualObjects,
    updateVisualObjects,
)
from scenes import buildSceneThreeBodies

from supaero2025.meshcat_viewer_wrapper import MeshcatVisualizer

# Build a scene
model, geom_model = buildSceneThreeBodies()
data = model.createData()
geom_data = geom_model.createData()

# %jupyter_snippet create
# Obtained by simply copying the collision model
visual_model = geom_model.copy()
preallocateVisualObjects(visual_model)

# Start meshcat
viz = MeshcatVisualizer(
    model=model, collision_model=geom_model, visual_model=visual_model
)
# %end_jupyter_snippet

# Force the collision margin to a huge value.
for r in geom_data.collisionRequests:
    r.security_margin = 10

# First isolated test
q = pin.randomConfiguration(model)
pin.computeCollisions(model, data, geom_model, geom_data, q, True)
contact_models = createContactModelsFromCollisions(model, data, geom_model, geom_data)
contact_datas = [cm.createData() for cm in contact_models]

# %jupyter_snippet display
updateVisualObjects(model, data, contact_models, contact_datas, visual_model, viz)
viz.display(q)
# %end_jupyter_snippet

# Start a random trajectory to display the witnesses.
v = (np.random.rand(model.nv) * 2 - 1) * 1e-3
r0 = [np.linalg.norm(q[7 * i : 7 * i + 3]) for i in range(model.nq // 7)]

# Place the patches based on distances or collisions? The visual result should
# be the same.
USE_DISTANCE = False

for t in range(100):
    q = pin.integrate(model, q, v * 10)
    for i in range(model.nq // 7):
        q[7 * i : 7 * i + 3] *= r0[i] / np.linalg.norm(q[7 * i : 7 * i + 3])

    if USE_DISTANCE:
        # The patches are on the contact surface
        pin.computeDistances(model, data, geom_model, geom_data, q)
        contact_models = createContactModelsFromDistances(
            model, data, geom_model, geom_data, 10
        )
    else:
        # With p3x, the patches are in between the two surfaces
        pin.computeCollisions(model, data, geom_model, geom_data, q)
        contact_models = createContactModelsFromCollisions(
            model, data, geom_model, geom_data
        )
    contact_datas = [cm.createData() for cm in contact_models]
    updateVisualObjects(model, data, contact_models, contact_datas, visual_model, viz)
    viz.display(q)

    time.sleep(0.01)

assert (
    len(visual_model.geometryObjects)
    == len(geom_model.geometryObjects) + 2 * COLPATCH_DEFAULT_PREALLOC
)
"""
