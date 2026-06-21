#include "c_space_mapper/configuration_mapper.hpp"
#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "c_space_mapper/footprint.hpp"
#include "c_space_mapper/cost_values.hpp"
#include "c_space_mapper_msgs/msg/configuration_space.hpp"

using namespace std;

namespace c_space_mapper
{
geometry_msgs::msg::Point make_point(double x, double y);
class MapperNode : public rclcpp::Node
{
public:
    MapperNode();
protected:
    void occupy2costmap(nav_msgs::msg::OccupancyGrid::SharedPtr occupymap, nav2_costmap_2d::Costmap2D & costmap_2d);
private:
    footprints fps_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr occupymap_subscriber_;
    rclcpp::Publisher<c_space_mapper_msgs::msg::ConfigurationSpace>::SharedPtr cspace_publisher_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr cost_ray_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cspace_vis_publisher_;
    void occupymapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);
    double a;
    double g_min;
    unsigned int g_divide_gmin;
    unsigned int crop_size;
    vector<string> footprints_strings_;

    // Cost ray parameters (configurable via params)
    double cost_ray_max_range_ = 5.0;
    double cost_ray_h_res_degrees_ = 10.0;
    unsigned int cost_ray_num_beams_ = 36;

    // C-space visualization parameters
    double cspace_vis_z_scale_ = 0.25;
    std::string robot_frame_id_ = "map";
};

}