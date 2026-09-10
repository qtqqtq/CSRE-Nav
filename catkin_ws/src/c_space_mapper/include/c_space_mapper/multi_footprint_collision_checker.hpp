#include "c_space_mapper/footprint_collision_checker.hpp"

#include <vector>

typedef std::vector<nav2_costmap_2d::Footprint> footprints;

namespace c_space_mapper
{

class MultiFootprintCollisionChecker
{
    public:
        MultiFootprintCollisionChecker();
        explicit MultiFootprintCollisionChecker(std::shared_ptr<nav2_costmap_2d::Costmap2D> costmap, const footprints & fps);
        double footprintCostAtPose(double x, double y, double theta);
    protected:
        nav2_costmap_2d::FootprintCollisionChecker<std::shared_ptr<nav2_costmap_2d::Costmap2D>> checker;
        footprints footprints_;
};

}