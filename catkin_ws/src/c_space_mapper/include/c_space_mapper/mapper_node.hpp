#pragma once
#include "c_space_mapper/configuration_mapper.hpp"
#include <vector>
#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Float32MultiArray.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include "c_space_mapper/footprint.hpp"
#include "c_space_mapper/cost_values.hpp"
#include <c_space_mapper_msgs/ConfigurationSpace.h>

namespace c_space_mapper
{

class MapperNode
{
public:
    MapperNode();

private:
    void occupymapCallback(const nav_msgs::OccupancyGrid::ConstPtr &msg);
    void odomCallback(const nav_msgs::Odometry::ConstPtr &msg);
    void occupy2costmap(const nav_msgs::OccupancyGrid::ConstPtr &occupy,
                        nav2_costmap_2d::Costmap2D &costmap_2d);

    footprints fps_;
    ros::Subscriber occupymap_sub_;
    ros::Subscriber odom_sub_;
    ros::Publisher  cspace_pub_;
    ros::Publisher  cost_ray_pub_;
    ros::Publisher  cost_ray_image_pub_;
    ros::Publisher  cspace_vis_pub_;
    ros::Publisher  cost_ray_samples_pub_;
    ros::Publisher  xy_projection_pub_;

    double a_ = 0.314;
    double g_min_ = 0.1;
    int g_divide_gmin_ = 2;
    int crop_size_ = 10;

    std::vector<int> scan_stepsizes_ = {2, 1};
    std::vector<int> scan_detection_windows_ = {10, 0};

    double cost_ray_max_range_ = 5.0;
    double cost_ray_h_res_degrees_ = 10.0;
    int    cost_ray_num_beams_ = 36;
    double cspace_vis_z_scale_ = 0.25;
    std::string robot_frame_id_ = "map";

    // Robot position from odometry (for cost_ray origin)
    double robot_x_ = 0.0;
    double robot_y_ = 0.0;
    bool has_odom_ = false;
};

} // namespace c_space_mapper
