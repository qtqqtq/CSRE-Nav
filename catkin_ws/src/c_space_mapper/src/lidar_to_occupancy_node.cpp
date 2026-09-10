#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <sensor_msgs/LaserScan.h>
#include <nav_msgs/OccupancyGrid.h>
#include <cmath>
#include <vector>
#include <cstring>

class LidarToOccupancyNode
{
public:
    LidarToOccupancyNode()
    {
        ros::NodeHandle pnh("~");

        pnh.param("map_resolution", map_res_, 0.05);
        pnh.param("map_width_m",    map_w_m_, 20.0);
        pnh.param("map_height_m",   map_h_m_, 20.0);
        pnh.param("origin_x",       origin_x_, -10.0);
        pnh.param("origin_y",       origin_y_, -10.0);
        pnh.param("min_range",      min_range_, 0.1);
        pnh.param("max_range",      max_range_, 20.0);
        pnh.param("ground_z_min",   gz_min_, 0.2);
        pnh.param("ground_z_max",   gz_max_, 2.0);
        pnh.param("sensor_offset_x", sensor_off_x_, 0.125);
        pnh.param("sensor_offset_y", sensor_off_y_, 0.0);
        pnh.param<std::string>("topic_in",  topic_in_, "/velodyne_points");
        pnh.param<std::string>("topic_out", topic_out_, "map_from_gazebo");
        pnh.param<std::string>("frame_id",  frame_id_, "map");
        pnh.param<std::string>("sensor_type", sensor_type_, "pointcloud2");

        w_ = static_cast<unsigned int>(std::ceil(map_w_m_ / map_res_));
        h_ = static_cast<unsigned int>(std::ceil(map_h_m_ / map_res_));
        size_t total = static_cast<size_t>(w_) * h_;
        occ_.resize(total, 0);

        ros::NodeHandle nh;
        if (sensor_type_ == "laserscan")
        {
            laser_sub_ = nh.subscribe(topic_in_, 1,
                                       &LidarToOccupancyNode::laserScanCallback, this);
            ROS_INFO("[lidar_to_occ] Subscribed to LaserScan: %s", topic_in_.c_str());
        }
        else
        {
            pc_sub_ = nh.subscribe(topic_in_, 1,
                                    &LidarToOccupancyNode::pointCloudCallback, this);
            ROS_INFO("[lidar_to_occ] Subscribed to PointCloud2: %s", topic_in_.c_str());
        }

        pub_ = nh.advertise<nav_msgs::OccupancyGrid>(topic_out_, 1, true); // latched

        ROS_INFO("[lidar_to_occ] %ux%u @ %.3fm  origin(%.1f,%.1f)  range[%.1f,%.1f]  sensor=%s",
                 w_, h_, map_res_, origin_x_, origin_y_, min_range_, max_range_,
                 sensor_type_.c_str());
    }

private:
    inline size_t idx(unsigned int x, unsigned int y) const { return static_cast<size_t>(y) * w_ + x; }

    bool worldToCell(double wx, double wy, int &cx, int &cy) const
    {
        cx = static_cast<int>(std::floor((wx - origin_x_) / map_res_));
        cy = static_cast<int>(std::floor((wy - origin_y_) / map_res_));
        return cx >= 0 && cx < static_cast<int>(w_) && cy >= 0 && cy < static_cast<int>(h_);
    }

    /// Bresenham line — mark every cell along (x0,y0)→(x1,y1) as free.
    void raycastFree(int x0, int y0, int x1, int y1, std::vector<int8_t> &occ)
    {
        int dx = std::abs(x1 - x0), dy = std::abs(y1 - y0);
        int sx = (x0 < x1) ? 1 : -1, sy = (y0 < y1) ? 1 : -1;
        int err = dx - dy;
        while (true) {
            // Stop before the endpoint — the endpoint is occupied
            if (x0 == x1 && y0 == y1) break;
            if (x0 >= 0 && x0 < static_cast<int>(w_) && y0 >= 0 && y0 < static_cast<int>(h_))
                occ[idx(x0, y0)] = 0;  // free
            int e2 = 2 * err;
            if (e2 > -dy) { err -= dy; x0 += sx; }
            if (e2 <  dx) { err += dx; y0 += sy; }
        }
    }

    void publish()
    {
        nav_msgs::OccupancyGrid msg;
        msg.header.stamp    = ros::Time::now();
        msg.header.frame_id = frame_id_;
        msg.info.resolution = static_cast<float>(map_res_);
        msg.info.width      = w_;
        msg.info.height     = h_;
        msg.info.origin.position.x = origin_x_;
        msg.info.origin.position.y = origin_y_;
        msg.info.origin.orientation.w = 1.0;
        msg.data = occ_;
        pub_.publish(msg);
    }

    // ---- PointCloud2 callback (Velodyne VLP-16) ----
    void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr &msg)
    {
        // Reset grid: default free (robot-centric local grid — unobserved = safe)
        std::fill(occ_.begin(), occ_.end(), 0);

        // Sensor origin in grid coordinates (base_link frame)
        int scx = static_cast<int>((sensor_off_x_ - origin_x_) / map_res_);
        int scy = static_cast<int>((sensor_off_y_ - origin_y_) / map_res_);

        sensor_msgs::PointCloud2ConstIterator<float> it_x(*msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> it_y(*msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> it_z(*msg, "z");

        for (; it_x != it_x.end(); ++it_x, ++it_y, ++it_z) {
            float px_s = *it_x, py_s = *it_y, pz_s = *it_z;
            double d = std::sqrt(px_s*px_s + py_s*py_s + pz_s*pz_s);
            if (d < min_range_ || d > max_range_) continue;
            if (pz_s < gz_min_ || pz_s > gz_max_) continue;

            // Transform from sensor frame to base_link
            float px = px_s + static_cast<float>(sensor_off_x_);
            float py = py_s + static_cast<float>(sensor_off_y_);

            int cx, cy;
            if (!worldToCell(px, py, cx, cy)) continue;

            // Mark hit cell as occupied
            occ_[idx(cx, cy)] = 100;

            // Raycast from sensor origin to hit: mark path as free
            raycastFree(scx, scy, cx, cy, occ_);
        }

        publish();
    }

    // ---- LaserScan callback (2D RPLIDAR / Hokuyo) ----
    void laserScanCallback(const sensor_msgs::LaserScan::ConstPtr &msg)
    {
        // Reset grid: default free (robot-centric local grid — unobserved = safe)
        std::fill(occ_.begin(), occ_.end(), 0);

        // Sensor origin in grid coordinates (base_link frame)
        int scx = static_cast<int>((sensor_off_x_ - origin_x_) / map_res_);
        int scy = static_cast<int>((sensor_off_y_ - origin_y_) / map_res_);

        float angle = msg->angle_min;
        for (size_t i = 0; i < msg->ranges.size(); ++i, angle += msg->angle_increment)
        {
            float r = msg->ranges[i];
            if (r < msg->range_min || r > msg->range_max) continue;
            if (r < min_range_ || r > max_range_) continue;

            // Transform from sensor frame to base_link
            double px = r * std::cos(angle) + sensor_off_x_;
            double py = r * std::sin(angle) + sensor_off_y_;

            int cx, cy;
            if (!worldToCell(px, py, cx, cy)) continue;

            // Mark hit cell as occupied
            occ_[idx(cx, cy)] = 100;

            // Raycast from sensor origin to hit: mark path as free
            raycastFree(scx, scy, cx, cy, occ_);
        }

        publish();
    }

    // params
    double map_res_, map_w_m_, map_h_m_, origin_x_, origin_y_;
    double min_range_, max_range_, gz_min_, gz_max_;
    double sensor_off_x_{0.125}, sensor_off_y_{0.0};
    std::string topic_in_, topic_out_, frame_id_, sensor_type_;
    unsigned int w_{0}, h_{0};

    // data
    std::vector<int8_t> occ_;

    ros::Subscriber pc_sub_;
    ros::Subscriber laser_sub_;
    ros::Publisher  pub_;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "lidar_to_occupancy_node");
    LidarToOccupancyNode node;
    ros::spin();
    return 0;
}
