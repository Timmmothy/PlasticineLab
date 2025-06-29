import taichi as ti

@ti.func
def length(x):
    return ti.sqrt(x.dot(x) + 1e-8)

@ti.func
def qrot3d(rot, v):
    # rot: vec4, p vec3
    qvec = ti.Vector([rot[1], rot[2], rot[3]])
    uv = qvec.cross(v)
    uuv = qvec.cross(uv)
    return v + 2 * (rot[0] * uv + uuv)

@ti.func
def qmul3d(q, r):
    terms = r.outer_product(q)
    w = terms[0, 0] - terms[1, 1] - terms[2, 2] - terms[3, 3]
    x = terms[0, 1] + terms[1, 0] - terms[2, 3] + terms[3, 2]
    y = terms[0, 2] + terms[1, 3] + terms[2, 0] - terms[3, 1]
    z = terms[0, 3] - terms[1, 2] + terms[2, 1] + terms[3, 0]
    out = ti.Vector([w, x, y, z])
    return out / ti.sqrt(out.dot(out)) # normalize it to prevent some unknown NaN problems.

@ti.func
def w2quat3d(axis_angle, dtype):
    #w = axis_angle.norm()
    w = ti.sqrt(axis_angle.dot(axis_angle) + 1e-16)
    out = ti.Vector.zero(dt=dtype, n=4)
    out[0] = 1.
    if w > 1e-7:
        v = (axis_angle/w) * ti.sin(w/2)
        #return ti.Vector([ti.cos(w/2), v * sin(w/2)])
        out[0] = ti.cos(w/2)
        out[1] = v[0]
        out[2] = v[1]
        out[3] = v[2]
    return out

@ti.func
def inv_trans3d(pos, position, rotation):
    #assert rotation.norm() > 0.9
    inv_quat = ti.Vector([rotation[0], -rotation[1], -rotation[2], -rotation[3]]).normalized()
    return qrot3d(inv_quat, pos - position)



# ----------------------------- 2d scenary ----------------------------------
@ti.func
def qrot2d(rot, v):
    return ti.Vector([rot[0]*v[0]-rot[1]*v[1], rot[1]*v[0] + rot[0]*v[1]])

@ti.func
def qmul2d(q, r):
    terms = r.outer_product(q)
    w = terms[0, 0] - terms[1, 1]
    x = terms[0, 1] + terms[1, 0]
    out = ti.Vector([w, x])
    return out / ti.sqrt(out.dot(out)) # normalize it to prevent some unknown NaN problems.

@ti.func
def inv_trans2d(pos, position, rotation):
    inv_quat = ti.Vector([rotation[0], -rotation[1]]).normalized()
    return qrot2d(inv_quat, pos - position)

@ti.func
def w2quat2d(axis_angle, dtype):
    w = ti.abs(axis_angle[0])
    out = ti.Vector.zero(dt=dtype, n=2)
    out[0] = 1.
    if w > 1e-9:
        out[0] = ti.cos(axis_angle[0])
        out[1] = ti.sin(axis_angle[0])
    return out
