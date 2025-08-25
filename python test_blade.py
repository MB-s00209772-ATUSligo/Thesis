import cadquery as cq

def test_geometry():
    # Hardcoded geometry inputs
    blade_length = 80
    blade_width = 30
    thickness = 5

    try:
        blade = (
            cq.Workplane("XY")
            .moveTo(0, 0)
            .lineTo(blade_width, 0)
            .lineTo(blade_width, blade_length)
            .lineTo(0, blade_length)
            .close()
            .extrude(thickness)
        )
        cq.exporters.export(blade, "test_blade.stl")
        print("✅ Blade geometry generated successfully.")
    except Exception as e:
        print("❌ Geometry failed:", e)

test_geometry()
