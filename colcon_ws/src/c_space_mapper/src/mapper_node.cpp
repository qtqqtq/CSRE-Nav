#include "c_space_mapper/mapper_node.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"

using namespace c_space_mapper;

geometry_msgs::msg::Point c_space_mapper::make_point(double x, double y)
{
    geometry_msgs::msg::Point p;
    p.x = x;
    p.y = y;
    p.z = 0.0;
    return p;
}


MapperNode::MapperNode() :Node("mapper_node")
{
    occupymap_subscriber_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
        "map_from_gazebo", 10,
        std::bind(&MapperNode::occupymapCallback, this, std::placeholders::_1));

    // Publisher for configuration space (existing)
    cspace_publisher_ = this->create_publisher<c_space_mapper_msgs::msg::ConfigurationSpace>(
        "configuration_space", 10);

    // Publisher for 3D cost rays (Float32MultiArray with layout: [num_beams, nz_])
    cost_ray_publisher_ = this->create_publisher<std_msgs::msg::Float32MultiArray>(
        "cost_rays", 10);

    // Publisher for 3D c-space visualization (PointCloud2 for RViz)
    cspace_vis_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "cspace_vis", 10);

    // ---- Declare parameters ----
    this->declare_parameter("a", (2*M_PI)/10);
    this->declare_parameter("g_min", 0.1);
    this->declare_parameter("g_divide_gmin", 2);
    this->declare_parameter("footprints",vector<string>{"[[0.0, 0.0], [0.2, 0.0], [0.2, 0.2], [0.0, 0.2]]"});
    this->declare_parameter("crop_size",10);

    // Cost ray parameters
    this->declare_parameter("cost_ray_max_range", 5.0);
    this->declare_parameter("cost_ray_h_res_degrees", 10.0);
    this->declare_parameter("cost_ray_num_beams", 36);

    // C-space visualization parameter
    this->declare_parameter("cspace_vis_z_scale", 0.25);
    this->declare_parameter("robot_frame_id", std::string("map"));

    // ---- Get parameters ----
    this->get_parameter("a", a);
    this->get_parameter("g_min", g_min);
    this->get_parameter("g_divide_gmin", g_divide_gmin);
    this->get_parameter("footprints", footprints_strings_);
    this->get_parameter("crop_size",crop_size);
    this->get_parameter("cost_ray_max_range", cost_ray_max_range_);
    this->get_parameter("cost_ray_h_res_degrees", cost_ray_h_res_degrees_);
    int num_beams_int;
    this->get_parameter("cost_ray_num_beams", num_beams_int);
    cost_ray_num_beams_ = static_cast<unsigned int>(num_beams_int);
    this->get_parameter("cspace_vis_z_scale", cspace_vis_z_scale_);
    this->get_parameter("robot_frame_id", robot_frame_id_);

    for (int i=0 ; i<footprints_strings_.size();i++){
        std::vector<geometry_msgs::msg::Point> fp;
        nav2_costmap_2d::makeFootprintFromString(footprints_strings_[i], fp);
        fps_.push_back(fp);
    }
}

void MapperNode::occupymapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
{
    nav2_costmap_2d::Costmap2D costmap_2d(
        msg->info.width,
        msg->info.height,
        msg->info.resolution,
        msg->info.origin.position.x,
        msg->info.origin.position.y,
        nav2_costmap_2d::FREE_SPACE);
    occupy2costmap(msg, costmap_2d);
    ConfigurationMapper mapper(
        std::make_shared<nav2_costmap_2d::Costmap2D>(costmap_2d),
        fps_,
        a,
        g_min,
        g_divide_gmin);
    mapper.scanRegion(crop_size, crop_size, -crop_size, -crop_size, {2, 1}, {10, 0});
    mapper.visualizeXYProjection("C-space XY Projection", 5);

    // ----- Publish Configuration Space -----
    auto cspace_msg = c_space_mapper_msgs::msg::ConfigurationSpace();
    cspace_msg.width = mapper.getXSize()-2*crop_size;
    cspace_msg.height = mapper.getYSize()-2*crop_size;
    cspace_msg.channels = mapper.getZSize();
    for (unsigned int i = crop_size; i < mapper.getXSize() - crop_size; i++)
    {
        for (unsigned int j = crop_size; j < mapper.getYSize() - crop_size; j++)
        {
            for (unsigned int k = 0; k < mapper.getZSize(); k++)
            {
                double cost = mapper.costAt(i, j, k);
                cspace_msg.data.push_back(static_cast<float>(cost));
            }
        }
    }
    cspace_publisher_->publish(cspace_msg);

    // ----- 3D Cost Ray Computation -----
    // Robot is at the center of the c-space grid in world coordinates.
    double center_x = mapper.getOriginX() + (mapper.getXSize() - 1) * mapper.getGridResolution() / 2.0;
    double center_y = mapper.getOriginY() + (mapper.getYSize() - 1) * mapper.getGridResolution() / 2.0;
    RCLCPP_INFO_ONCE(this->get_logger(),
        "C-space center (robot world): (%.3f, %.3f), origin: (%.3f, %.3f)",
        center_x, center_y, mapper.getOriginX(), mapper.getOriginY());
    double h_res = cost_ray_h_res_degrees_ * M_PI / 180.0;

    auto cost_rays = mapper.computeCostRays3D(
        center_x, center_y,
        h_res, cost_ray_num_beams_,
        cost_ray_max_range_);

    auto cost_ray_msg = std_msgs::msg::Float32MultiArray();
    cost_ray_msg.layout.dim.resize(2);
    cost_ray_msg.layout.dim[0].label = "beam";
    cost_ray_msg.layout.dim[0].size = cost_ray_num_beams_;
    cost_ray_msg.layout.dim[0].stride = mapper.getZSize();
    cost_ray_msg.layout.dim[1].label = "theta";
    cost_ray_msg.layout.dim[1].size = mapper.getZSize();
    cost_ray_msg.layout.dim[1].stride = 1;
    cost_ray_msg.layout.data_offset = 0;
    cost_ray_msg.data = std::move(cost_rays);
    cost_ray_publisher_->publish(cost_ray_msg);

    // ----- 3D C-Space Visualization PointCloud2 -----
    // Collect occupied voxels from the c-space
    std::vector<CspaceVoxel> voxels;
    mapper.collectOccupiedVoxels(voxels, crop_size, cspace_vis_z_scale_, center_x, center_y);

    // Build PointCloud2 message
    auto cloud_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
    cloud_msg->header.stamp = this->now();
    cloud_msg->header.frame_id = robot_frame_id_;
    cloud_msg->height = 1;               // unordered point cloud
    cloud_msg->width = voxels.size();
    cloud_msg->is_dense = true;

    // Define fields: x, y, z, intensity
    cloud_msg->fields.resize(4);
    cloud_msg->fields[0].name = "x";
    cloud_msg->fields[0].offset = 0;
    cloud_msg->fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
    cloud_msg->fields[0].count = 1;
    cloud_msg->fields[1].name = "y";
    cloud_msg->fields[1].offset = 4;
    cloud_msg->fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
    cloud_msg->fields[1].count = 1;
    cloud_msg->fields[2].name = "z";
    cloud_msg->fields[2].offset = 8;
    cloud_msg->fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
    cloud_msg->fields[2].count = 1;
    cloud_msg->fields[3].name = "intensity";
    cloud_msg->fields[3].offset = 12;
    cloud_msg->fields[3].datatype = sensor_msgs::msg::PointField::FLOAT32;
    cloud_msg->fields[3].count = 1;
    cloud_msg->point_step = 16;  // 4 floats × 4 bytes
    cloud_msg->row_step = cloud_msg->point_step * voxels.size();
    cloud_msg->data.resize(cloud_msg->row_step);

    // Fill point cloud data using iterators
    sensor_msgs::PointCloud2Iterator<float> iter_x(*cloud_msg, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(*cloud_msg, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(*cloud_msg, "z");
    sensor_msgs::PointCloud2Iterator<float> iter_intensity(*cloud_msg, "intensity");

    float nz_f = static_cast<float>(mapper.getZSize());
    for (const auto &v : voxels)
    {
        *iter_x = v.x;
        *iter_y = v.y;
        *iter_z = v.z;
        // intensity = normalized theta index (0..1), so RViz can color by orientation
        float theta_idx = (nz_f > 0.0f) ? (v.z / static_cast<float>(cspace_vis_z_scale_)) / nz_f : 0.0f;
        *iter_intensity = theta_idx;

        ++iter_x;
        ++iter_y;
        ++iter_z;
        ++iter_intensity;
    }

    cspace_vis_publisher_->publish(*cloud_msg);

    // Log summary
    RCLCPP_INFO_ONCE(this->get_logger(),
        "Published cspace_vis: %zu occupied voxels, z_scale=%.3f",
        voxels.size(), cspace_vis_z_scale_);
}

void MapperNode::occupy2costmap(nav_msgs::msg::OccupancyGrid::SharedPtr occupymap,nav2_costmap_2d::Costmap2D & costmap_2d)
{
    unsigned int cell_size_x = occupymap->info.width;
    unsigned int cell_size_y = occupymap->info.height;
    for (int i =0;i< cell_size_x; i++)
    {
        for (int j =0;j< cell_size_y; j++)
        {
            unsigned int index = j*cell_size_x + i;
            int8_t occupancy_value = occupymap->data[index];
            unsigned char cost_value;
            if (occupancy_value == 100 or occupancy_value == 50) // Occupied
                cost_value = nav2_costmap_2d::LETHAL_OBSTACLE;
            else if(occupancy_value == 0) // Free
                cost_value = nav2_costmap_2d::FREE_SPACE;
            costmap_2d.setCost(i, j, cost_value);
        }
    }
}

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MapperNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}