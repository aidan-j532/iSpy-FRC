// Robot-frame -> viewer-frame axis mapping, supplied by the unit pack
// (viewer.robot_to_viewer). The default keeps the historical FRC mapping
// (x, y, z) -> (x, z, -y).
const DEFAULT_AXIS_MAP = ["x", "z", "-y"];

const AXIS_VECTORS = {
  x: [1, 0, 0],
  y: [0, 1, 0],
  z: [0, 0, 1],
};

function axisExpression(expr) {
  if (typeof expr !== "string") return null;
  let text = expr.trim().toLowerCase().replace(/\s+/g, "");
  if (!text) return null;
  let sign = 1;
  if (text[0] === "+" || text[0] === "-") {
    sign = text[0] === "-" ? -1 : 1;
    text = text.slice(1);
  }
  const vec = AXIS_VECTORS[text];
  if (!vec) return null;
  return [sign * vec[0], sign * vec[1], sign * vec[2]];
}

function determinant3(m) {
  return (
    m[0] * (m[4] * m[8] - m[5] * m[7]) -
    m[1] * (m[3] * m[8] - m[5] * m[6]) +
    m[2] * (m[3] * m[7] - m[4] * m[6])
  );
}

export function axisMapMatrix(axisMap) {
  const rows = Array.isArray(axisMap) && axisMap.length === 3 ? axisMap : null;
  const parsed = rows ? rows.map(axisExpression) : null;
  if (!parsed || parsed.some((row) => row === null)) {
    return axisMapMatrix(DEFAULT_AXIS_MAP);
  }
  const flat = parsed.flat();
  // Only proper rotations (det +1) keep quaternions valid.
  if (Math.abs(determinant3(flat) - 1) > 1e-9) {
    return axisMapMatrix(DEFAULT_AXIS_MAP);
  }
  return flat;
}

function transpose3(m) {
  return [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]];
}

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

function mapPoint(matrix, x, y, z) {
  return [
    matrix[0] * x + matrix[1] * y + matrix[2] * z,
    matrix[3] * x + matrix[4] * y + matrix[5] * z,
    matrix[6] * x + matrix[7] * y + matrix[8] * z,
  ];
}

export function robotToViewer(x, y, z, axisMap) {
  return mapPoint(axisMapMatrix(axisMap), x, y, z);
}

export function robotToViewerQuat(roll, pitch, yaw, axisMap) {
  const R_robot = matMul(matRz(yaw), matMul(matRy(pitch), matRx(roll)));
  // The viewer has always applied the inverse of the position mapping to
  // orientations (the historical [x,z,-y] viewer is Rx(+90) here), so use the
  // transpose to keep existing scenes and orientation tests unchanged.
  const M = transpose3(axisMapMatrix(axisMap));
  return matToQuat(matMul(M, R_robot));
}
