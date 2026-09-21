// shared robot <-> three.js viewer transform. the robot frame is +X right,
// +Y forward, +Z up (that's what every vision pipeline ships). the three.js
// scene is +X right, +Y up, +Z out of screen, so going robot -> viewer is the
// proper rotation (x, y, z) -> (x, z, -y). don't swap two axes here, that
// flips handedness and folds tilted detections into the wrong plane.

// rotation matrix helpers; column-vector convention like three.js. row-major
// flattened ([r00,r01,r02, r10,r11,r12, r20,r21,r22]) so index math is boring.
const M = [1, 0, 0, 0, 0, 1, 0, -1, 0]; // Rx(-90) rotated col-major: (x,y,z)->(x,z,-y); +Y->-Z, +Z->+Y, +X stays

function matRx(a) {
  const c = Math.cos(a), s = Math.sin(a);
  return [1, 0, 0, 0, c, -s, 0, s, c];
}
function matRy(a) {
  const c = Math.cos(a), s = Math.sin(a);
  return [c, 0, s, 0, 1, 0, -s, 0, c];
}
function matRz(a) {
  const c = Math.cos(a), s = Math.sin(a);
  return [c, -s, 0, s, c, 0, 0, 0, 1];
}
function matMul(A, B) {
  return [
    A[0]*B[0] + A[1]*B[3] + A[2]*B[6], A[0]*B[1] + A[1]*B[4] + A[2]*B[7], A[0]*B[2] + A[1]*B[5] + A[2]*B[8],
    A[3]*B[0] + A[4]*B[3] + A[5]*B[6], A[3]*B[1] + A[4]*B[4] + A[5]*B[7], A[3]*B[2] + A[4]*B[5] + A[5]*B[8],
    A[6]*B[0] + A[7]*B[3] + A[8]*B[6], A[6]*B[1] + A[7]*B[4] + A[8]*B[7], A[6]*B[2] + A[7]*B[5] + A[8]*B[8],
  ];
}

// Shepperd's method: 3x3 rotation -> quaternion [x, y, z, w].
function matToQuat(R) {
  const tr = R[0] + R[4] + R[8];
  if (tr > 0) {
    const s = Math.sqrt(tr + 1) * 2;
    return [(R[7] - R[5]) / s, (R[2] - R[6]) / s, (R[3] - R[1]) / s, s / 4];
  }
  if (R[0] > R[4] && R[0] > R[8]) {
    const s = Math.sqrt(1 + R[0] - R[4] - R[8]) * 2;
    return [s / 4, (R[1] + R[3]) / s, (R[2] + R[6]) / s, (R[7] - R[5]) / s];
  }
  if (R[4] > R[8]) {
    const s = Math.sqrt(1 + R[4] - R[0] - R[8]) * 2;
    return [(R[1] + R[3]) / s, s / 4, (R[5] + R[7]) / s, (R[2] - R[6]) / s];
  }
  const s = Math.sqrt(1 + R[8] - R[0] - R[4]) * 2;
  return [(R[2] + R[6]) / s, (R[5] + R[7]) / s, s / 4, (R[3] - R[1]) / s];
}

// robot (x,y,z) -> three.js (x,z,-y). positions and kpt clouds both use this.
export function robotToViewer(x, y, z) {
  return [x, z, -y];
}

// object roll/pitch/yaw in robot frame -> viewer quaternion [x,y,z,w].
// the pipeline eulers reconstruct R = Rz(yaw)*Ry(pitch)*Rx(roll), the exact
// order _matrix_to_euler decodes in april_tag, so compose that with M.
export function robotToViewerQuat(roll, pitch, yaw) {
  const R_robot = matMul(matRz(yaw), matMul(matRy(pitch), matRx(roll)));
  return matToQuat(matMul(M, R_robot));
}