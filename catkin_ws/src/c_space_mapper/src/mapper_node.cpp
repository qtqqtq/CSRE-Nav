#include "c_space_mapper/mapper_node.hpp"
#include <deque>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc.hpp>

namespace c_space_mapper
{

MapperNode::MapperNode()
{
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    // ---- Read parameters ----
    pnh.param("a", a_, (2.0 * M_PI) / 10.0);
    pnh.param("g_min", g_min_, 0.1);
    pnh.param("g_divide_gmin", g_divide_gmin_, 1);
    pnh.param("crop_size", crop_size_, 10);
    pnh.param("scan_stepsizes", scan_stepsizes_, std::vector<int>{2, 1});
    pnh.param("scan_detection_windows", scan_detection_windows_, std::vector<int>{10, 0});

    std::vector<std::string> fp_strs;
    pnh.getParam("footprints", fp_strs);
    if (fp_strs.empty())
        fp_strs.push_back("[[0.0, 0.0], [0.2, 0.0], [0.2, 0.2], [0.0, 0.2]]");

    for (const auto &s : fp_strs) {
        std::vector<geometry_msgs::Point> fp;
        nav2_costmap_2d::makeFootprintFromString(s, fp);
        fps_.push_back(fp);
    }

    pnh.param("cost_ray_max_range", cost_ray_max_range_, 5.0);
    pnh.param("cost_ray_h_res_degrees", cost_ray_h_res_degrees_, 10.0);
    pnh.param("cost_ray_num_beams", cost_ray_num_beams_, 36);
    pnh.param("cspace_vis_z_scale", cspace_vis_z_scale_, 0.25);
    pnh.param<std::string>("robot_frame_id", robot_frame_id_, "map");

    ROS_INFO("[mapper_node] a=%.4f  g_min=%.2f  crop=%d  footprints=%zu",
             a_, g_min_, crop_size_, fps_.size());

    // ---- Subscribers / Publishers ----
    occupymap_sub_ = nh.subscribe("map_from_gazebo", 1,
                                   &MapperNode::occupymapCallback, this);
    odom_sub_ = nh.subscribe("/r1/odom", 1,
                             &MapperNode::odomCallback, this);
    cspace_pub_  = nh.advertise<c_space_mapper_msgs::ConfigurationSpace>(
        "configuration_space", 10);
    cost_ray_pub_ = nh.advertise<std_msgs::Float32MultiArray>("cost_rays", 10);
    cost_ray_image_pub_ = nh.advertise<sensor_msgs::Image>("cost_ray_image", 10);
    cspace_vis_pub_ = nh.advertise<sensor_msgs::PointCloud2>("cspace_vis", 10);
    cost_ray_samples_pub_ = nh.advertise<sensor_msgs::PointCloud2>("cost_ray_samples", 10);
    xy_projection_pub_ = nh.advertise<sensor_msgs::Image>("xy_projection_image", 10);
}

void MapperNode::odomCallback(const nav_msgs::Odometry::ConstPtr &msg)
{
    robot_x_ = msg->pose.pose.position.x;
    robot_y_ = msg->pose.pose.position.y;
    has_odom_ = true;
}

void MapperNode::occupymapCallback(const nav_msgs::OccupancyGrid::ConstPtr &msg)
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
        fps_, a_, g_min_, g_divide_gmin_);

    std::deque<unsigned int> stepsizes(scan_stepsizes_.begin(), scan_stepsizes_.end());
    std::deque<unsigned int> detection_windows(scan_detection_windows_.begin(), scan_detection_windows_.end());
    mapper.scanRegion(crop_size_, crop_size_,
                      -static_cast<int>(crop_size_),
                      -static_cast<int>(crop_size_),
                      stepsizes, detection_windows);

    // --- Publish ConfigurationSpace ---
    c_space_mapper_msgs::ConfigurationSpace cspace_msg;
    cspace_msg.width  = mapper.getXSize() - 2 * crop_size_;
    cspace_msg.height = mapper.getYSize() - 2 * crop_size_;
    cspace_msg.channels = mapper.getZSize();
    for (unsigned int i = crop_size_; i < mapper.getXSize() - crop_size_; ++i)
        for (unsigned int j = crop_size_; j < mapper.getYSize() - crop_size_; ++j)
            for (unsigned int k = 0; k < mapper.getZSize(); ++k)
                cspace_msg.data.push_back(static_cast<float>(mapper.costAt(i, j, k)));
    cspace_pub_.publish(cspace_msg);

    // --- Cost Rays ---
    // Occupancy grid is built robot-centrically (LiDAR points in sensor frame
    // are used as-is, without transforming to a global frame).  The robot is
    // therefore always at (0, 0) in the costmap / c-space.
    double cx = 0.0;
    double cy = 0.0;
    double h_res = cost_ray_h_res_degrees_ * M_PI / 180.0;

    auto rays = mapper.computeCostRays3D(cx, cy, h_res,
                                         static_cast<unsigned int>(cost_ray_num_beams_),
                                         cost_ray_max_range_);

    // --- Cost Ray Image (grayscale: black=far/free, white=near/obstacle) ---
    {
        unsigned int n_beams = static_cast<unsigned int>(cost_ray_num_beams_);
        unsigned int n_theta = mapper.getZSize();

        // Unwrap so the robot front (beam 0, +x) sits at the image center; the
        // back (180°) then lands at both the left and right edges.
        unsigned int shift = n_beams / 2;

        cv::Mat gray(static_cast<int>(n_theta), static_cast<int>(n_beams), CV_8UC1);
        for (unsigned int beam = 0; beam < n_beams; ++beam)
        {
            unsigned int col = (beam + shift) % n_beams;
            for (unsigned int k = 0; k < n_theta; ++k)
            {
                float dist = rays[beam * n_theta + k];
                // near=white, far=black
                float norm = 1.0f - std::min(dist, static_cast<float>(cost_ray_max_range_))
                                     / static_cast<float>(cost_ray_max_range_);
                gray.at<uchar>(static_cast<int>(k), static_cast<int>(col)) =
                    static_cast<uchar>(255.0f * norm);
            }
        }

        // Resize for visibility
        cv::Mat scaled;
        cv::resize(gray, scaled, cv::Size(static_cast<int>(n_beams) * 4,
                                          static_cast<int>(n_theta) * 8),
                   0, 0, cv::INTER_NEAREST);

        sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(
            std_msgs::Header(), "mono8", scaled).toImageMsg();
        img_msg->header.stamp    = ros::Time::now();
        img_msg->header.frame_id = robot_frame_id_;
        cost_ray_image_pub_.publish(img_msg);
    }

    std_msgs::Float32MultiArray ray_msg;
    ray_msg.layout.dim.resize(2);
    ray_msg.layout.dim[0].label  = "beam";
    ray_msg.layout.dim[0].size   = static_cast<unsigned int>(cost_ray_num_beams_);
    ray_msg.layout.dim[0].stride = mapper.getZSize();
    ray_msg.layout.dim[1].label  = "theta";
    ray_msg.layout.dim[1].size   = mapper.getZSize();
    ray_msg.layout.dim[1].stride = 1;
    ray_msg.data = std::move(rays);
    cost_ray_pub_.publish(ray_msg);

    // --- PointCloud2 visualization ---
    std::vector<CspaceVoxel> voxels;
    mapper.collectOccupiedVoxels(voxels, crop_size_, cspace_vis_z_scale_, 0.0, 0.0);

    sensor_msgs::PointCloud2 cloud;
    cloud.header.stamp    = ros::Time::now();
    cloud.header.frame_id = robot_frame_id_;
    cloud.height = 1;
    cloud.width  = voxels.size();
    cloud.is_dense = true;

    cloud.fields.resize(4);
    int off = 0;
    for (int i = 0; i < 4; ++i) {
        cloud.fields[i].datatype = sensor_msgs::PointField::FLOAT32;
        cloud.fields[i].count    = 1;
        cloud.fields[i].offset   = off;
        off += 4;
    }
    cloud.fields[0].name = "x";
    cloud.fields[1].name = "y";
    cloud.fields[2].name = "z";
    cloud.fields[3].name = "intensity";
    cloud.point_step = 16;
    cloud.row_step   = cloud.point_step * voxels.size();
    cloud.data.resize(cloud.row_step);

    sensor_msgs::PointCloud2Iterator<float> it_x(cloud, "x");
    sensor_msgs::PointCloud2Iterator<float> it_y(cloud, "y");
    sensor_msgs::PointCloud2Iterator<float> it_z(cloud, "z");
    sensor_msgs::PointCloud2Iterator<float> it_i(cloud, "intensity");

    float nz_f = static_cast<float>(mapper.getZSize());
    for (const auto &v : voxels) {
        *it_x = v.x;  ++it_x;
        *it_y = v.y;  ++it_y;
        *it_z = v.z;  ++it_z;
        *it_i = (nz_f > 0.0f) ? (v.z / static_cast<float>(cspace_vis_z_scale_)) / nz_f : 0.0f;
        ++it_i;
    }
    cspace_vis_pub_.publish(cloud);

    // --- Cost ray sampling points (visualization) ---
    std::vector<CostRaySample> samples;
    mapper.collectCostRaySamples(cx, cy, h_res,
                                 static_cast<unsigned int>(cost_ray_num_beams_),
                                 cost_ray_max_range_, /*theta_idx=*/0, samples);

    sensor_msgs::PointCloud2 samples_cloud;
    samples_cloud.header.stamp    = ros::Time::now();
    samples_cloud.header.frame_id = robot_frame_id_;
    samples_cloud.height = 1;
    samples_cloud.width  = samples.size();
    samples_cloud.is_dense = true;

    samples_cloud.fields.resize(4);
    int soff = 0;
    for (int i = 0; i < 4; ++i) {
        samples_cloud.fields[i].datatype = sensor_msgs::PointField::FLOAT32;
        samples_cloud.fields[i].count    = 1;
        samples_cloud.fields[i].offset   = soff;
        soff += 4;
    }
    samples_cloud.fields[0].name = "x";
    samples_cloud.fields[1].name = "y";
    samples_cloud.fields[2].name = "z";
    samples_cloud.fields[3].name = "intensity";
    samples_cloud.point_step = 16;
    samples_cloud.row_step   = samples_cloud.point_step * samples.size();
    samples_cloud.data.resize(samples_cloud.row_step);

    sensor_msgs::PointCloud2Iterator<float> sx(samples_cloud, "x");
    sensor_msgs::PointCloud2Iterator<float> sy(samples_cloud, "y");
    sensor_msgs::PointCloud2Iterator<float> sz(samples_cloud, "z");
    sensor_msgs::PointCloud2Iterator<float> si(samples_cloud, "intensity");

    for (const auto &p : samples) {
        *sx = p.x;    ++sx;
        *sy = p.y;    ++sy;
        *sz = 0.0f;   ++sz;
        *si = p.dist; ++si;
    }
    cost_ray_samples_pub_.publish(samples_cloud);

    // --- XY projection image (θ folded into RGB) ---
    {
        cv::Mat xy_img = mapper.getXYProjectionImage(/*scale=*/2);
        if (!xy_img.empty()) {
            sensor_msgs::ImagePtr xy_msg = cv_bridge::CvImage(
                std_msgs::Header(), "bgr8", xy_img).toImageMsg();
            xy_msg->header.stamp    = ros::Time::now();
            xy_msg->header.frame_id = robot_frame_id_;
            xy_projection_pub_.publish(xy_msg);
        }
    }

    ROS_INFO_ONCE("[mapper_node] C-space center: (%.2f, %.2f)  published %zu voxels",
                  cx, cy, voxels.size());
}

void MapperNode::occupy2costmap(const nav_msgs::OccupancyGrid::ConstPtr &occupy,
                                 nav2_costmap_2d::Costmap2D &costmap_2d)
{
    unsigned int w = occupy->info.width;
    unsigned int h = occupy->info.height;
    for (unsigned int x = 0; x < w; ++x) {
        for (unsigned int y = 0; y < h; ++y) {
            int8_t v = occupy->data[y * w + x];
            unsigned char c;
            if (v == 100 || v == 50)
                c = nav2_costmap_2d::LETHAL_OBSTACLE;
            else if (v == 0)
                c = nav2_costmap_2d::FREE_SPACE;
            else
                c = nav2_costmap_2d::NO_INFORMATION;
            costmap_2d.setCost(x, y, c);
        }
    }
}

} // namespace c_space_mapper
