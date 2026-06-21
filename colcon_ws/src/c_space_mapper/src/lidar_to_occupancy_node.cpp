#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "std_msgs/msg/header.hpp"

#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <cmath>
#include <vector>
#include <cstring>

using namespace std::chrono_literals;

class LidarToOccupancyNode : public rclcpp::Node
{
public:
    LidarToOccupancyNode()
    : Node("lidar_to_occupancy_node")
    {
        // ---- Declare parameters ----
        this->declare_parameter("map_resolution", 0.05);
        this->declare_parameter("map_width_m", 20.0);
        this->declare_parameter("map_height_m", 20.0);
        this->declare_parameter("origin_x", -10.0);
        this->declare_parameter("origin_y", -10.0);
        this->declare_parameter("min_range", 0.1);
        this->declare_parameter("max_range", 30.0);
        this->declare_parameter("ground_z_min", -0.05);
        this->declare_parameter("ground_z_max", 2.0);  // filter out ground, keep obstacles
        this->declare_parameter("publish_rate", 10.0);
        this->declare_parameter("topic_in", "/velodyne_points");
        this->declare_parameter("topic_out", "map_from_gazebo");
        this->declare_parameter("frame_id", "map");
        this->declare_parameter("num_accumulate_scans", 10);  // accumulate N scans before clearing

        // ---- Get parameters ----
        this->get_parameter("map_resolution", map_res_);
        this->get_parameter("map_width_m", map_width_m_);
        this->get_parameter("map_height_m", map_height_m_);
        this->get_parameter("origin_x", origin_x_);
        this->get_parameter("origin_y", origin_y_);
        this->get_parameter("min_range", min_range_);
        this->get_parameter("max_range", max_range_);
        this->get_parameter("ground_z_min", ground_z_min_);
        this->get_parameter("ground_z_max", ground_z_max_);
        this->get_parameter("publish_rate", publish_rate_);
        this->get_parameter("topic_in", topic_in_);
        this->get_parameter("topic_out", topic_out_);
        this->get_parameter("frame_id", frame_id_);
        this->get_parameter("num_accumulate_scans", num_accumulate_scans_);

        // ---- Compute grid dimensions ----
        map_width_cells_  = static_cast<unsigned int>(std::ceil(map_width_m_  / map_res_));
        map_height_cells_ = static_cast<unsigned int>(std::ceil(map_height_m_ / map_res_));
        size_t total_cells = static_cast<size_t>(map_width_cells_) * map_height_cells_;
        occupancy_grid_.resize(total_cells, 0);       // 0 = unknown, will be used as FREE initially
        hit_counter_.resize(total_cells, 0);            // count hits per cell
        miss_counter_.resize(total_cells, 0);           // count miss (free) per cell

        RCLCPP_INFO(this->get_logger(),
            "Grid: %u×%u cells @ %.3fm res, origin (%.1f, %.1f), range [%.1f, %.1f]m",
            map_width_cells_, map_height_cells_, map_res_, origin_x_, origin_y_,
            min_range_, max_range_);

        // ---- Subscriber ----
        sub_pcl_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            topic_in_, rclcpp::SensorDataQoS(),
            std::bind(&LidarToOccupancyNode::cloudCallback, this, std::placeholders::_1));

        // ---- Publisher (latched so mapper_node can pick it up at any time) ----
        pub_og_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(topic_out_,
            rclcpp::QoS(rclcpp::KeepLast(1)).transient_local());

        // ---- Timer for periodic publish ----
        double period = 1.0 / publish_rate_;
        timer_ = this->create_wall_timer(
            std::chrono::duration<double>(period),
            std::bind(&LidarToOccupancyNode::publishMap, this));

        scan_count_ = 0;
    }

private:
    // ---- Member variables ----
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcl_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr pub_og_;
    rclcpp::TimerBase::SharedPtr timer_;

    std::string topic_in_;
    std::string topic_out_;
    std::string frame_id_;

    double map_res_;
    double map_width_m_;
    double map_height_m_;
    double origin_x_;
    double origin_y_;
    double min_range_;
    double max_range_;
    double ground_z_min_;
    double ground_z_max_;
    double publish_rate_;
    unsigned int map_width_cells_;
    unsigned int map_height_cells_;
    unsigned int num_accumulate_scans_;

    // Persistent occupancy map
    std::vector<int8_t> occupancy_grid_;
    std::vector<unsigned int> hit_counter_;
    std::vector<unsigned int> miss_counter_;
    unsigned int scan_count_;

    // ---- Index helpers ----
    inline size_t cellIndex(unsigned int x, unsigned int y) const
    {
        return static_cast<size_t>(y) * map_width_cells_ + x;
    }

    inline bool worldToCell(double wx, double wy, int &cx, int &cy) const
    {
        double fx = (wx - origin_x_) / map_res_;
        double fy = (wy - origin_y_) / map_res_;
        cx = static_cast<int>(std::floor(fx));
        cy = static_cast<int>(std::floor(fy));
        return (cx >= 0 && cx < static_cast<int>(map_width_cells_) &&
                cy >= 0 && cy < static_cast<int>(map_height_cells_));
    }

    // ---- Raycasting to mark free space (Bresenham 2D) ----
    void raycastFree(int x0, int y0, int x1, int y1)
    {
        int dx = std::abs(x1 - x0);
        int dy = std::abs(y1 - y0);
        int sx = (x0 < x1) ? 1 : -1;
        int sy = (y0 < y1) ? 1 : -1;
        int err = dx - dy;

        while (true)
        {
            if (x0 >= 0 && x0 < static_cast<int>(map_width_cells_) &&
                y0 >= 0 && y0 < static_cast<int>(map_height_cells_))
            {
                size_t idx = cellIndex(static_cast<unsigned int>(x0), static_cast<unsigned int>(y0));
                miss_counter_[idx]++;
            }

            if (x0 == x1 && y0 == y1) break;

            int e2 = 2 * err;
            if (e2 > -dy) { err -= dy; x0 += sx; }
            if (e2 <  dx) { err += dx; y0 += sy; }
        }
    }

    // ---- Point cloud callback ----
    void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        // Accumulate scans. After num_accumulate_scans_, merge into occupancy_grid_ and reset.
        scan_count_++;

        // Get origin of the LiDAR (sensor position) from the header. If not available,
        // assume (0,0) in the map frame. The header typically has the sensor pose if
        // the point cloud is in sensor frame.
        // For simplicity, we treat the sensor as at origin (map origin assumption).
        // A more complete implementation would look up the TF transform.

        // Convert sensor origin to grid cell
        int sensor_cx = static_cast<int>((0.0 - origin_x_) / map_res_);  // rough center
        int sensor_cy = static_cast<int>((0.0 - origin_y_) / map_res_);

        // Iterate over points
        sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");

        for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z)
        {
            float px = *iter_x;
            float py = *iter_y;
            float pz = *iter_z;

            // Range filter
            double dist = std::sqrt(px*px + py*py + pz*pz);
            if (dist < min_range_ || dist > max_range_)
                continue;

            // Height filter: keep points above ground_z_min and below ground_z_max
            if (pz < ground_z_min_ || pz > ground_z_max_)
                continue;

            // Project to grid
            // Note: px,py,pz are in the point cloud's frame (typically base_link or velodyne frame).
            // For Gazebo simulation with static robot, this is usually close to map frame.
            // A proper implementation would transform using TF.
            int cx, cy;
            if (!worldToCell(static_cast<double>(px), static_cast<double>(py), cx, cy))
                continue;

            // Mark occupied cell
            size_t idx = cellIndex(static_cast<unsigned int>(cx), static_cast<unsigned int>(cy));
            hit_counter_[idx]++;

            // Raycast from sensor to this point to mark free space
            raycastFree(sensor_cx, sensor_cy, cx, cy);
        }

        // After accumulating enough scans, merge and reset
        if (scan_count_ >= num_accumulate_scans_)
        {
            mergeAndReset();
            scan_count_ = 0;
        }
    }

    void mergeAndReset()
    {
        for (size_t i = 0; i < occupancy_grid_.size(); ++i)
        {
            unsigned int hits = hit_counter_[i];
            unsigned int misses = miss_counter_[i];
            unsigned int total = hits + misses;

            if (total == 0)
            {
                // No observation — keep as unknown (-1)
                occupancy_grid_[i] = -1;
                continue;
            }

            // Simple log-odds update: if hits > misses, mark occupied.
            // More hits than misses → likely occupied.
            if (hits > misses)
                occupancy_grid_[i] = 100;  // occupied
            else if (misses > 0)
                occupancy_grid_[i] = 0;    // free
            else
                occupancy_grid_[i] = -1;   // unknown
        }

        // Reset counters
        std::fill(hit_counter_.begin(), hit_counter_.end(), 0);
        std::fill(miss_counter_.begin(), miss_counter_.end(), 0);
    }

    // ---- Publish map ----
    void publishMap()
    {
        auto msg = nav_msgs::msg::OccupancyGrid();
        msg.header.stamp = this->now();
        msg.header.frame_id = frame_id_;
        msg.info.resolution = static_cast<float>(map_res_);
        msg.info.width = map_width_cells_;
        msg.info.height = map_height_cells_;
        msg.info.origin.position.x = origin_x_;
        msg.info.origin.position.y = origin_y_;
        msg.info.origin.position.z = 0.0;
        msg.info.origin.orientation.w = 1.0;
        msg.data = occupancy_grid_;

        pub_og_->publish(msg);
    }
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LidarToOccupancyNode>());
    rclcpp::shutdown();
    return 0;
}
