#include <cmath>
#include <opencv2/core.hpp>
#include "c_space_mapper/multi_footprint_collision_checker.hpp"
#include <std_msgs/Float32MultiArray.h>
#include <string>
#include <utility>
#include <deque>
#include <vector>

namespace c_space_mapper
{

// Lightweight struct for 3D visualization voxel data
struct CspaceVoxel
{
    float x;
    float y;
    float z;
    float cost;
};

// Lightweight struct for cost-ray sample-point visualization
struct CostRaySample
{
    float x;             // world position of the sample point
    float y;
    float dist;          // distance along the beam (meters)
    unsigned int beam;   // beam index
};

class ConfigurationMapper
{
    public:
        explicit ConfigurationMapper(std::shared_ptr<nav2_costmap_2d::Costmap2D> costmap, const footprints & fps,double a=(2*M_PI)/200,double g_min=0.025,unsigned int g_divide_gmin=2);
        std::array<double,3> configurationToMap(unsigned int i,unsigned int j,unsigned int k);
        bool isInCollision(unsigned int i,unsigned int j,unsigned int k);
        void checkPosition(unsigned int i,unsigned int j);
        bool checkPositionWithBoundary(unsigned int i,unsigned int j,unsigned int detection_window=3);
        void scanRegion(unsigned int i_start,unsigned int j_start,int i_end=-1,int j_end=-1,std::deque<unsigned int> stepsizes={2,1},std::deque<unsigned int> detection_windows={4,2});
        void visualizeXYProjection(const std::string &window_name, unsigned int scale);

        // Build the XY projection image (θ folded into RGB channels), scaled by
        // `scale` for visibility. Returns an empty Mat if the c-space is empty.
        // Reused both by the interactive window and by image publishing.
        cv::Mat getXYProjectionImage(unsigned int scale) const;

        void visualizecostmap2d();
        std::shared_ptr<std_msgs::Float32MultiArray> getMultiArray();
        double costAt(unsigned int i,unsigned int j,unsigned int k) const;

        // ---- 3D Cost Ray Methods (NavRL-style) ----

        // Convert world coordinates to grid indices. Returns false if out of bounds.
        bool worldToGrid(double wx, double wy, unsigned int &i, unsigned int &j) const;

        // Check if a grid position is fully blocked (ALL θ orientations are in collision).
        bool isPositionBlocked(unsigned int i, unsigned int j) const;

        // Cast a single ray through the 3D c-space (x, y, θ) at a fixed θ orientation.
        // Steps along (dir_x, dir_y) in the x-y plane, checking costAt at the given theta_idx.
        // Returns distance to first collision, or max_range if no collision found.
        double castCostRay3D(double start_x, double start_y,
                             double dir_x, double dir_y,
                             unsigned int theta_idx, double max_range) const;

        // Compute 3D cost rays in a LiDAR-like 360° pattern.
        // For each horizontal beam direction AND each robot orientation θ,
        // casts a ray through (x,y) space and returns distance to first collision.
        // This mirrors NavRL's H_beams × V_beams 3D scan structure:
        //   - Horizontal: sweep 360° with h_res angular resolution (num_beams rays)
        //   - θ dimension: for each beam, sweep all nz_ robot orientations
        //
        // Returns flat vector of size num_beams * nz_, beam-major layout:
        //   index = beam_idx * nz_ + theta_idx
        // Each entry is the distance to the first (x,y,θ) voxel where costAt > 0.5,
        // or max_range if no collision along the ray.
        //
        // center_x, center_y: robot position in world coords (c-space center assumed)
        // h_res: angular resolution between horizontal beams (radians)
        // num_beams: number of horizontal beams
        // max_range: maximum ray length
        std::vector<float> computeCostRays3D(double center_x, double center_y,
                                             double h_res, unsigned int num_beams,
                                             double max_range) const;

        // Collect occupied voxels for 3D RViz visualization.
        // Each voxel where cost > 0.5 is collected as a CspaceVoxel.
        // Coordinates are offset by (offset_x, offset_y) so the robot is at origin.
        // z = theta_idx * z_scale. Only voxels within [crop, dim-crop) are included.
        void collectOccupiedVoxels(std::vector<CspaceVoxel> &voxels, unsigned int crop_size,
                                   double z_scale,
                                   double offset_x = 0.0, double offset_y = 0.0) const;

        // Collect the discrete sample points along every cost-ray beam, for a
        // single robot orientation theta_idx. Mirrors castCostRay3D's 3D DDA
        // traversal: θ sweeps linearly from 0 to θ_k over the ray, stepping
        // (i, j, k) cells, up to max_range or the first collision. Samples are
        // 2D (x, y) points; θ only affects the stopping point.
        void collectCostRaySamples(double center_x, double center_y,
                                   double h_res, unsigned int num_beams, double max_range,
                                   unsigned int theta_idx,
                                   std::vector<CostRaySample> &samples) const;

        // Accessors
        unsigned int getXSize() const { return nx_; }
        unsigned int getYSize() const { return ny_; }
        unsigned int getZSize() const { return nz_; }
        double getGridResolution() const { return _g_min; }
        double getOriginX() const { return _origin_x; }
        double getOriginY() const { return _origin_y; }
        double getSizeXMeters() const { return _size_x; }
        double getSizeYMeters() const { return _size_y; }
    protected:
        MultiFootprintCollisionChecker fpsChecker;
        double _a;
        double _g_min;
        unsigned int _g_divide_gmin;
        double _origin_x;
        double _origin_y;
        double _g;
        double _size_x;
        double _size_y;
        std::vector<float> configuration_space;
        unsigned int nx_ = 0;
        unsigned int ny_ = 0;
        unsigned int nz_ = 0;

    private:
        inline size_t idx(unsigned int i, unsigned int j, unsigned int k) const
        {
            return static_cast<size_t>(i) * ny_ * nz_ + static_cast<size_t>(j) * nz_ + k;
        }
};

}