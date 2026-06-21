#include "c_space_mapper/multi_footprint_collision_checker.hpp"

namespace c_space_mapper
{   

MultiFootprintCollisionChecker::MultiFootprintCollisionChecker()
{
}

MultiFootprintCollisionChecker::MultiFootprintCollisionChecker(
    std::shared_ptr<nav2_costmap_2d::Costmap2D> costmap,
    const footprints & fps)
: checker(costmap),
footprints_(fps)
{
}

double MultiFootprintCollisionChecker::footprintCostAtPose(
    double x, double y, double theta)
{
    double result = 0;
    for(nav2_costmap_2d::Footprint ft : footprints_)
    {
        if(checker.footprintCostAtPose(x,y,theta,ft) > result)
            result = checker.footprintCostAtPose(x,y,theta,ft);
    }
    return result;
}

}