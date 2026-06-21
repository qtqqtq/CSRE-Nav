#include <stdio.h>
#include <cmath>
#include <cassert>
#include <iostream>

#include "c_space_mapper/cost_values.hpp"
#include "c_space_mapper/footprint_collision_checker.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "c_space_mapper/configuration_mapper.hpp"

geometry_msgs::msg::Point make_point(double x, double y, double z = 0.0)
{
  geometry_msgs::msg::Point p;
  p.x = x;
  p.y = y;
  p.z = z;
  return p;
}

using namespace c_space_mapper;

#define TEST(name) printf("\n=== %s ===\n", name)
#define PASS() printf("  PASS\n")
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("  FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); return 1; } \
} while(0)
#define CHECK_FLOAT_EQ(a, b, tol, msg) do { \
    if (std::fabs((a) - (b)) > (tol)) { \
        printf("  FAIL: %s  expected=%f  actual=%f\n", msg, (double)(b), (double)(a)); \
        return 1; \
    } \
} while(0)

// Helper: build a 20m×20m map (400×400 cells @ 0.05m)
nav2_costmap_2d::Costmap2D make_large_map()
{
    return nav2_costmap_2d::Costmap2D(400, 400, 0.05, 0.0, 0.0, nav2_costmap_2d::FREE_SPACE);
}

int test_basic_cspace()
{
    TEST("Basic C-Space Construction & scanRegion");

    auto test_map = make_large_map();
    // Wall at costmap col 160 = world x=8.0, spanning y [2, 8]
    for (int row = 40; row <= 160; row++) {
        test_map.setCost(160, row, nav2_costmap_2d::LETHAL_OBSTACLE);
    }

    nav2_costmap_2d::Footprint tiny_fp = {
        make_point(0.01, 0.01), make_point(0.01, -0.01),
        make_point(-0.01, -0.01), make_point(-0.01, 0.01)
    };
    footprints test_footprints = {tiny_fp};
    auto test_map_ptr = std::make_shared<nav2_costmap_2d::Costmap2D>(test_map);

    ConfigurationMapper mapper(test_map_ptr, test_footprints);

    unsigned int nx = mapper.getXSize();
    unsigned int ny = mapper.getYSize();
    unsigned int nz = mapper.getZSize();
    printf("  C-space dimensions: nx=%u  ny=%u  nz=%u\n", nx, ny, nz);
    CHECK(nx > 0 && ny > 0 && nz > 0, "dimensions should be positive");

    // Scan with step=1 for full coverage
    mapper.scanRegion(0, 0, nx, ny, {1}, {0});

    // Wall at costmap col 160 → world x=8.0 → c-space grid i = 8.0/0.025 = 320
    // Row 100 → world y=5.0 → c-space grid j = 5.0/0.025 = 200
    double cost_near_wall = mapper.costAt(320, 200, 0);
    // Free space far from wall
    double cost_free = mapper.costAt(80, 80, 0);
    printf("  costAt near wall grid(320,200,0): %f\n", cost_near_wall);
    printf("  costAt free space grid(80,80,0): %f\n", cost_free);
    CHECK(cost_near_wall > 0.5,  "cost near wall should indicate collision");
    CHECK(cost_free < 0.5,       "cost in free space should indicate free");

    PASS();
    return 0;
}

int test_world_to_grid()
{
    TEST("worldToGrid");

    auto test_map = make_large_map();
    nav2_costmap_2d::Footprint tiny_fp = {
        make_point(0.01, 0.01), make_point(0.01, -0.01),
        make_point(-0.01, -0.01), make_point(-0.01, 0.01)
    };
    footprints fps = {tiny_fp};
    auto map_ptr = std::make_shared<nav2_costmap_2d::Costmap2D>(test_map);
    ConfigurationMapper mapper(map_ptr, fps);

    unsigned int i, j;

    CHECK(mapper.worldToGrid(0.0, 0.0, i, j), "origin should be in grid");
    CHECK(i == 0 && j == 0, "origin -> (0,0)");

    // 20m / 0.025 = 800, +1 = 801 cells. Grid idx 800 = world 20.0
    CHECK(mapper.worldToGrid(19.99, 19.99, i, j), "(19.99, 19.99) should be in grid");
    CHECK(mapper.worldToGrid(-0.1, 5.0, i, j) == false, "negative x should be out");
    CHECK(mapper.worldToGrid(5.0, 20.1, i, j) == false, "y beyond map should be out");
    CHECK(mapper.worldToGrid(10.0, 10.0, i, j), "center (10,10) should be in grid");

    PASS();
    return 0;
}

int test_cost_ray_3d_basic()
{
    TEST("castCostRay3D — open space vs wall");

    auto test_map = make_large_map();
    // Wall at costmap col 160 = world x=8.0, spanning y [2, 8]
    for (int row = 40; row <= 160; row++) {
        test_map.setCost(160, row, nav2_costmap_2d::LETHAL_OBSTACLE);
    }

    nav2_costmap_2d::Footprint tiny_fp = {
        make_point(0.01, 0.01), make_point(0.01, -0.01),
        make_point(-0.01, -0.01), make_point(-0.01, 0.01)
    };
    footprints fps = {tiny_fp};
    auto map_ptr = std::make_shared<nav2_costmap_2d::Costmap2D>(test_map);
    ConfigurationMapper mapper(map_ptr, fps);
    mapper.scanRegion(0, 0, mapper.getXSize(), mapper.getYSize(), {1}, {0});

    double max_range = 4.0;
    unsigned int theta_0 = 0;

    // Center at world (10, 5) — 5m from north/south boundaries, 10m from east/west
    double cx = 10.0;
    double cy = 5.0;
    printf("  Center: (%f, %f)\n", cx, cy);

    // West (-1, 0): from x=10 to wall at x=8 → hit at ~2.0m
    double d_west = mapper.castCostRay3D(cx, cy, -1.0, 0.0, theta_0, max_range);
    printf("  Ray WEST: %f m (wall at x=8, expect ~2.0)\n", d_west);
    CHECK_FLOAT_EQ(d_west, 2.0, 0.15, "west should hit wall at ~2.0m");

    // East (1, 0): from x=10 to boundary x=20 → 10m open space → max_range
    double d_east = mapper.castCostRay3D(cx, cy, 1.0, 0.0, theta_0, max_range);
    printf("  Ray EAST: %f m (expect max_range=%f)\n", d_east, max_range);
    CHECK_FLOAT_EQ(d_east, max_range, 1e-5, "east should be max_range (open)");

    // North (0, 1): from y=5 to boundary y=20 → 15m open → max_range
    double d_north = mapper.castCostRay3D(cx, cy, 0.0, 1.0, theta_0, max_range);
    printf("  Ray NORTH: %f m (expect max_range)\n", d_north);
    CHECK_FLOAT_EQ(d_north, max_range, 1e-5, "north should be max_range (open)");

    // Different theta indices should return similar results for the same geometry
    unsigned int theta_mid = mapper.getZSize() / 2;
    double d_west_tmid = mapper.castCostRay3D(cx, cy, -1.0, 0.0, theta_mid, max_range);
    printf("  Ray WEST theta=%u: %f m\n", theta_mid, d_west_tmid);
    CHECK_FLOAT_EQ(d_west_tmid, d_west, 0.15, "same ray, different theta should be similar");

    PASS();
    return 0;
}

int test_cost_ray_theta_axis()
{
    TEST("castCostRay3D — theta axis distinguishes orientations");

    // Use a wall that blocks only some θ orientations.
    // Place obstacle at a specific (x,y) and verify that costAt reflects the θ-dependent collision.
    auto test_map = make_large_map();
    // Small obstacle at costmap (200, 200) = world (10, 10)
    test_map.setCost(200, 200, nav2_costmap_2d::LETHAL_OBSTACLE);

    nav2_costmap_2d::Footprint tiny_fp = {
        make_point(0.01, 0.01), make_point(0.01, -0.01),
        make_point(-0.01, -0.01), make_point(-0.01, 0.01)
    };
    footprints fps = {tiny_fp};
    auto map_ptr = std::make_shared<nav2_costmap_2d::Costmap2D>(test_map);
    ConfigurationMapper mapper(map_ptr, fps);
    mapper.scanRegion(0, 0, mapper.getXSize(), mapper.getYSize(), {1}, {0});

    // costmap cell (200,200) -> c-space grid (400, 400)
    // All theta should be in collision at the obstacle cell
    for (unsigned int k = 0; k < mapper.getZSize(); ++k) {
        double c = mapper.costAt(400, 400, k);
        CHECK(c > 0.5, "obstacle grid cell should be collision for all theta");
    }
    printf("  All %u theta at obstacle grid(400,400) are collision: OK\n", mapper.getZSize());

    // Nearby free cell grid(380, 400) = world (9.5, 10.0) — just west of obstacle
    // With tiny footprint, theta=0 (facing east) should NOT collide with the obstacle
    // (robot at 9.5 facing east, footprint extends to 9.51, obstacle at 10.0)
    // But with tiny footprint radius 0.014, at grid(380, 400) = (9.5, 10.0),
    // the distance to obstacle at (10.0, 10.0) is 0.5m > footprint size → free.
    double c_free = mapper.costAt(380, 400, 0);
    printf("  costAt free cell grid(380,400,0): %f\n", c_free);
    CHECK(c_free < 0.5, "nearby free cell should be free");

    PASS();
    return 0;
}

int test_compute_cost_rays_3d()
{
    TEST("computeCostRays3D — full LiDAR pattern");

    auto test_map = make_large_map();
    // Wall at costmap col 140 = world x=7.0, spanning y [2, 8]
    for (int row = 40; row <= 160; row++) {
        test_map.setCost(140, row, nav2_costmap_2d::LETHAL_OBSTACLE);
    }

    nav2_costmap_2d::Footprint tiny_fp = {
        make_point(0.01, 0.01), make_point(0.01, -0.01),
        make_point(-0.01, -0.01), make_point(-0.01, 0.01)
    };
    footprints fps = {tiny_fp};
    auto map_ptr = std::make_shared<nav2_costmap_2d::Costmap2D>(test_map);
    ConfigurationMapper mapper(map_ptr, fps);
    mapper.scanRegion(0, 0, mapper.getXSize(), mapper.getYSize(), {1}, {0});

    // Center at (10, 5). Wall at x=7 → west hits at 3.0m. max_range=5 → west hits.
    double cx = 10.0;
    double cy = 5.0;
    unsigned int num_beams = 8;
    double h_res = 2.0 * M_PI / num_beams;
    double max_range = 5.0;
    unsigned int nz = mapper.getZSize();

    auto rays = mapper.computeCostRays3D(cx, cy, h_res, num_beams, max_range);
    size_t expected_size = static_cast<size_t>(num_beams) * nz;
    printf("  Rays size: %zu (num_beams=%u × nz=%u)\n", rays.size(), num_beams, nz);
    CHECK(rays.size() == expected_size, "cost rays size should be num_beams * nz");

    // Verify beam-major layout consistency: all theta within a beam have same distance
    for (unsigned int beam = 0; beam < num_beams; ++beam) {
        double angle = beam * h_res * 180.0 / M_PI;
        double d0 = rays[beam * nz + 0];
        // Show first 4 theta values
        printf("  Beam %u (%3.0f°): ", beam, angle);
        for (unsigned int k = 0; k < 4 && k < nz; ++k)
            printf("%5.2f ", rays[beam * nz + k]);
        printf("\n");

        // All theta in same beam should return same distance (same (x,y) geometry)
        for (unsigned int k = 1; k < nz; ++k) {
            double dk = rays[beam * nz + k];
            CHECK_FLOAT_EQ(dk, d0, 0.02, "same beam, all theta should have same distance");
        }
    }
    printf("  All theta within each beam consistent: OK\n");

    // Beam 4 (180° = west): should hit wall at ~3.0m
    unsigned int beam_west = num_beams / 2;
    double d_west = rays[beam_west * nz + 0];
    printf("  West beam distance: %f m (wall at x=7, expect ~3.0)\n", d_west);
    CHECK(d_west > 0.5 && d_west < max_range - 0.01, "west beam should hit wall");

    // Beam 0 (0° = east): should be max_range (open)
    double d_east = rays[0 * nz + 0];
    printf("  East beam distance: %f m (expect max_range=%f)\n", d_east, max_range);
    CHECK_FLOAT_EQ(d_east, max_range, 1e-5, "east beam should be max_range");

    PASS();
    return 0;
}

int test_is_position_blocked()
{
    TEST("isPositionBlocked");

    auto test_map = make_large_map();
    // Obstacle at costmap cell (200, 200) = world (10, 10)
    test_map.setCost(200, 200, nav2_costmap_2d::LETHAL_OBSTACLE);

    nav2_costmap_2d::Footprint tiny_fp = {
        make_point(0.01, 0.01), make_point(0.01, -0.01),
        make_point(-0.01, -0.01), make_point(-0.01, 0.01)
    };
    footprints fps = {tiny_fp};
    auto map_ptr = std::make_shared<nav2_costmap_2d::Costmap2D>(test_map);
    ConfigurationMapper mapper(map_ptr, fps);
    mapper.scanRegion(0, 0, mapper.getXSize(), mapper.getYSize(), {1}, {0});

    // Grid cell for world (10, 10): 10/0.025 = 400
    bool blocked_at_obstacle = mapper.isPositionBlocked(400, 400);
    printf("  isPositionBlocked at obstacle: %d\n", blocked_at_obstacle);
    CHECK(blocked_at_obstacle, "obstacle position should be fully blocked");

    // Grid cell for world (5, 5): 5/0.025 = 200 — far from obstacle AND far from boundary
    bool blocked_at_free = mapper.isPositionBlocked(200, 200);
    printf("  isPositionBlocked at free space: %d\n", blocked_at_free);
    CHECK(!blocked_at_free, "free space should not be blocked");

    PASS();
    return 0;
}

int test_center_position()
{
    TEST("Center position calculation");

    auto test_map = make_large_map();
    nav2_costmap_2d::Footprint tiny_fp = {
        make_point(0.01, 0.01), make_point(0.01, -0.01),
        make_point(-0.01, -0.01), make_point(-0.01, 0.01)
    };
    footprints fps = {tiny_fp};
    auto map_ptr = std::make_shared<nav2_costmap_2d::Costmap2D>(test_map);
    ConfigurationMapper mapper(map_ptr, fps);

    double cx = mapper.getOriginX() + (mapper.getXSize() - 1) * mapper.getGridResolution() / 2.0;
    double cy = mapper.getOriginY() + (mapper.getYSize() - 1) * mapper.getGridResolution() / 2.0;
    printf("  Origin: (%f, %f),  Size: %u×%u,  Res: %f\n",
           mapper.getOriginX(), mapper.getOriginY(),
           mapper.getXSize(), mapper.getYSize(), mapper.getGridResolution());
    printf("  Center: (%f, %f)\n", cx, cy);

    // 400 cells × 0.05m = 20m. C-space: ceil(20/0.025)+1 = 801 cells.
    // center = 0 + (801-1)*0.025/2 = 800*0.025/2 = 10.0
    CHECK_FLOAT_EQ(cx, 10.0, 1e-5, "center x should be 10.0");
    CHECK_FLOAT_EQ(cy, 10.0, 1e-5, "center y should be 10.0");

    unsigned int i, j;
    CHECK(mapper.worldToGrid(cx, cy, i, j), "center should be in grid");

    PASS();
    return 0;
}

int main()
{
    printf("==========================================\n");
    printf("  C-Space Mapper — 3D Cost Ray Unit Tests\n");
    printf("==========================================\n");

    int failures = 0;
    failures += test_basic_cspace();
    failures += test_world_to_grid();
    failures += test_cost_ray_3d_basic();
    failures += test_cost_ray_theta_axis();
    failures += test_compute_cost_rays_3d();
    failures += test_is_position_blocked();
    failures += test_center_position();

    printf("\n==========================================\n");
    if (failures == 0) {
        printf("  ALL TESTS PASSED\n");
    } else {
        printf("  %d TEST(S) FAILED\n", failures);
    }
    printf("==========================================\n");

    return failures;
}