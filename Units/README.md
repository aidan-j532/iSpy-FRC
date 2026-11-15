# Unit packs

A unit pack is a single JSON file in this directory that describes everything
about a measurement convention:

- how values typed by a user become iSpy's internal inches
- how internal inches become values published to NetworkTables, the API, and
  the 3D viewer
- axis signs for published positions and orientations
- short labels shown in the UI
- default mount angles for a new camera
- 3D viewer axis mapping, bounds, scale sliders, grid, and camera

The built-in packs are `frc.json`, `metric.json`, and `imperial.json`. Any
`*.json` file in this directory is loaded at startup. A file with the same
`key` as a built-in pack overrides that pack; a new `key` adds a new option to
the Settings, Setup Wizard, camera, and add-on pages. `POST /api/units/upload`
and `POST /api/units/write` validate and install a pack at runtime.

## Schema

```json
{
  "schema_version": 2,
  "key": "frc",
  "name": "FRC (inches in, meters out)",
  "description": "Shown in the unit picker.",
  "input_unit": "inch",
  "output_unit": "meter",
  "input_to_inches": 1.0,
  "inches_to_output": 0.0254,
  "aliases": ["wpilib"],
  "axis": {
    "x": 1.0,
    "y": 1.0,
    "z": 1.0,
    "roll": 1.0,
    "pitch": 1.0,
    "yaw": 1.0
  },
  "labels": {
    "input": "in",
    "output": "m",
    "speed": "m/s"
  },
  "defaults": {
    "camera_pitch_deg": 0.0
  },
  "viewer": {
    "robot_to_viewer": ["x", "z", "-y"],
    "camera_roll_offset_deg": 0.0,
    "camera_pitch_offset_deg": 0.0,
    "camera_yaw_offset_deg": 0.0,
    "fog_density": 0.04,
    "axes_size": 8.0,
    "bounds": {
      "x": { "min": -10.0, "max": 10.0 },
      "y": { "min": -10.0, "max": 10.0 },
      "z": { "min": 0.0, "max": 5.0 }
    },
    "scale": {
      "x": { "min": -10.0, "max": 10.0, "step": 0.1, "default": 1.0 },
      "y": { "min": -10.0, "max": 10.0, "step": 0.1, "default": 1.0 },
      "z": { "min": -10.0, "max": 10.0, "step": 0.1, "default": 1.0 }
    },
    "grid": {
      "extent": 20.0,
      "divisions": 80,
      "marker_size": 0.25
    },
    "camera": {
      "fov": 60.0,
      "near": 0.1,
      "far": 200.0,
      "position": [9.0, 9.0, 9.0],
      "target": [0.0, 0.0, 0.0]
    }
  }
}
```

### Top level

| Key | Meaning |
| --- | --- |
| `schema_version` | Schema revision. Currently `2`; unknown versions still load with defaults. |
| `key` | Unique pack id. Used by `Config/config.json`'s `unit` value and `/api/units/select`. |
| `name`, `description` | Human-readable text for the unit pickers. |
| `input_unit`, `output_unit` | Physical units a user types and the pack publishes. Informational plus fallback matching when a pack is looked up by unit name. |
| `input_to_inches` | Multiply a user-entered value by this to get inches. |
| `inches_to_output` | Multiply internal inches by this to get the published value. |
| `aliases` | Extra names that resolve to this pack, such as `wpilib`. |

A pack is resolved by, in order: `key`, an entry in `aliases`, matching
`input_unit`, then matching `output_unit`.

### `axis`

Sign multiplier applied to each published component. `1.0` leaves the axis
alone; `-1.0` flips it. Positions use `x`/`y`/`z`; triangulated and PnP
orientations use `roll`/`pitch`/`yaw`.

Do not flip the physical frame itself: the pipelines, calibration, and
triangulation always work in `+X right`, `+Y forward`, `+Z up`, positive yaw
right, positive pitch down. `axis` only changes how values are published.

### `labels`

Short strings for the UI. If a label is omitted it is derived from
`input_unit`/`output_unit`.

### `defaults`

| Key | Meaning |
| --- | --- |
| `camera_pitch_deg` | Pitch assigned to a newly created camera. `0` is level and looks forward. |

### `viewer`

| Key | Meaning |
| --- | --- |
| `robot_to_viewer` | Three signed robot axes that feed the viewer's X, Y, and Z axes. Must be a signed permutation with determinant `+1`, for example `["x", "z", "-y"]`. |
| `camera_roll_offset_deg`, `camera_pitch_offset_deg`, `camera_yaw_offset_deg` | Extra rotation applied to the 3D viewer's camera overlay. Normally `0`: the camera's physical `yaw`/`pitch` already define its orientation. |
| `fog_density` | Three.js exponential fog density. Scale it with the pack: a scene measured in inches needs a much smaller value than one measured in meters. `0` disables fog. |
| `axes_size` | Length of the solid X/Y/Z axis helper in viewer units. Scale it with `grid.extent`. |
| `bounds.x/y/z` | Published-unit bounds used to frame the scene. `viewer_settings()` also derives the same bounds in viewer coordinates. |
| `scale.x/y/z` | Min, max, step, and reset value for the viewer's per-axis scale sliders. |
| `grid.extent` | Width/depth of the floor grid in viewer units. If omitted it is derived from `bounds`. |
| `grid.divisions` | Number of grid lines. |
| `grid.marker_size` | Size of detection/cube markers and the nominal planar marker size. |
| `camera.fov`, `camera.near`, `camera.far` | Three.js perspective camera settings. |
| `camera.position`, `camera.target` | Default orbit position and target. |

## Adding a pack

1. Copy an existing file so every field starts in a known state.
2. Change `key`, `name`, `description`, the two conversion factors, aliases, and
   labels.
3. Adjust `viewer` if the published scale is very different (the imperial pack
   uses larger bounds and camera distances, for example).
4. Restart iSpy, or upload the file from **Add-ons → Units**.

`GET /api/units` returns the normalized packs and the active key. The 3D viewer
also embeds the active pack's normalized viewer settings in `/viewer3d`.
