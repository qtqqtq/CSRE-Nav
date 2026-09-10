#include "c_space_mapper/configuration_mapper.hpp"
#include "c_space_mapper/cost_values.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <limits>

namespace c_space_mapper
{

ConfigurationMapper::ConfigurationMapper(
    std::shared_ptr<nav2_costmap_2d::Costmap2D> costmap,
    const footprints & fps,
    double a,
    double g_min,
    unsigned int g_divide_gmin):
    fpsChecker(costmap, fps),
    _a(a),
    _g_min(g_min),
    _g_divide_gmin(g_divide_gmin),
    _origin_x(costmap->getOriginX()),
    _origin_y(costmap->getOriginY()),
    _size_x(costmap->getSizeInMetersX()),
    _size_y(costmap->getSizeInMetersY()),
    _g(_g_min * _g_divide_gmin)
{
    nx_ = static_cast<unsigned int>(std::ceil(_size_x / _g_min)) + 1;
    ny_ = static_cast<unsigned int>(std::ceil(_size_y / _g_min)) + 1;
    nz_ = static_cast<unsigned int>(2 * M_PI / _a) + 1;
    configuration_space.resize(static_cast<size_t>(nx_) * ny_ * nz_, 0.0f);
}

double ConfigurationMapper::costAt(unsigned int i, unsigned int j, unsigned int k) const
{
    if (i >= nx_ || j >= ny_ || k >= nz_)
        throw std::out_of_range("Configuration indices out of range");
    return static_cast<double>(configuration_space[idx(i, j, k)]);
}

std::array<double,3> ConfigurationMapper::configurationToMap(unsigned int i,unsigned int j,unsigned int k)
{
    std::array<double,3> result;
    result[0] = _origin_x + i*_g_min;
    result[1] = _origin_y + j*_g_min;
    result[2] = k*_a;
    return result;
}

bool ConfigurationMapper::isInCollision(unsigned int i,unsigned int j,unsigned int k)
{
    if (i >= nx_ || j >= ny_)
        throw std::out_of_range("Configuration indices out of range");
    auto [x,y,theta] = configurationToMap(i,j,k);
    return fpsChecker.footprintCostAtPose(x,y,theta) >= nav2_costmap_2d::LETHAL_OBSTACLE;
}

void ConfigurationMapper::checkPosition(unsigned int i,unsigned int j)
{
    for (unsigned int k = 0; k < nz_; k++)
    {
        if (isInCollision(i, j, k))
            configuration_space[idx(i, j, k)] = 1.0f;
        else
            configuration_space[idx(i, j, k)] = 0.0f;
    }
}

bool ConfigurationMapper::checkPositionWithBoundary(unsigned int i,unsigned int j,unsigned int detection_window)
{
    int collision_count = 0;
    for (unsigned int k = 0; k < nz_; k++)
    {
        if (isInCollision(i, j, k))
        {
            configuration_space[idx(i, j, k)] = 1.0f;
            collision_count++;
        }
        else
        {
            configuration_space[idx(i, j, k)] = 0.0f;
        }
    }

    // Check if this (i,j) position is near a boundary: there exists a theta k
    // where k is free (0) but has collisions (1) within the detection window
    // on both sides (circularly).
    bool isboundary = false;
    int nz = static_cast<int>(nz_);
    for (int k = 0; k < nz; ++k)
    {
        if (configuration_space[idx(i, j, static_cast<unsigned int>(k))] > 0.5f)
            continue;

        // Check detection_window before k (circular)
        bool has_collision_before = false;
        for (int d = 1; d <= static_cast<int>(detection_window); ++d)
        {
            int kk = (k - d) % nz;
            if (kk < 0) kk += nz;
            if (configuration_space[idx(i, j, static_cast<unsigned int>(kk))] > 0.5f)
            {
                has_collision_before = true;
                break;
            }
        }

        if (!has_collision_before) continue;

        // Check detection_window after k (circular)
        bool has_collision_after = false;
        for (int d = 1; d <= static_cast<int>(detection_window); ++d)
        {
            int kk = (k + d) % nz;
            if (configuration_space[idx(i, j, static_cast<unsigned int>(kk))] > 0.5f)
            {
                has_collision_after = true;
                break;
            }
        }

        if (has_collision_after)
        {
            isboundary = true;
            break;
        }
    }
    return isboundary;
}

void ConfigurationMapper::scanRegion(unsigned int i_start,unsigned int j_start,int i_end,int j_end,std::deque<unsigned int> stepsizes,std::deque<unsigned int> detection_windows)
{
    if(stepsizes.empty() || detection_windows.empty())
        return;
    auto stepsize = stepsizes.front();
    stepsizes.pop_front();
    auto detection_window = detection_windows.front();
    detection_windows.pop_front();
    if (i_end < 0)
        i_end = static_cast<int>(nx_) + i_end;
    if (j_end < 0)
        j_end = static_cast<int>(ny_) + j_end;
    if (static_cast<unsigned int>(i_end) > nx_ || static_cast<unsigned int>(j_end) > ny_)
        throw std::out_of_range("Scan region indices out of range");
    for(unsigned int i = i_start; i < static_cast<unsigned int>(i_end); i+=stepsize)
    {
        for(unsigned int j = j_start; j < static_cast<unsigned int>(j_end); j+=stepsize)
        {
            if(detection_window != 0){
                bool near_boundary = checkPositionWithBoundary(i, j, detection_window);
                if(near_boundary){
                    ConfigurationMapper::scanRegion(
                        i-stepsize/2>=i_start?i-stepsize/2:i_start,
                        j-stepsize/2>=j_start?j-stepsize/2:j_start,
                        i+stepsize/2+1<static_cast<unsigned int>(i_end)?static_cast<int>(i)+static_cast<int>(stepsize)/2+1:static_cast<int>(i_end)-1,
                        j+stepsize/2+1<static_cast<unsigned int>(j_end)?static_cast<int>(j)+static_cast<int>(stepsize)/2+1:static_cast<int>(j_end)-1,
                        stepsizes,
                        detection_windows
                    );
                }
            }
            else{
                checkPosition(i, j);
            }
        }
    }
}

std::shared_ptr<std_msgs::Float32MultiArray> ConfigurationMapper::getMultiArray()
{
    auto multi_array = std::make_shared<std_msgs::Float32MultiArray>();

    multi_array->layout.dim.resize(3);
    multi_array->layout.dim[0].label = "x";
    multi_array->layout.dim[0].size = nx_;
    multi_array->layout.dim[0].stride = ny_ * nz_;
    multi_array->layout.dim[1].label = "y";
    multi_array->layout.dim[1].size = ny_;
    multi_array->layout.dim[1].stride = nz_;
    multi_array->layout.dim[2].label = "theta";
    multi_array->layout.dim[2].size = nz_;
    multi_array->layout.dim[2].stride = 1;
    multi_array->layout.data_offset = 0;

    multi_array->data = configuration_space;

    return multi_array;
}

// ============================================================================
// 3D Cost Ray Methods (NavRL-style)
// ============================================================================

bool ConfigurationMapper::worldToGrid(double wx, double wy, unsigned int &i, unsigned int &j) const
{
    double fi = (wx - _origin_x) / _g_min;
    double fj = (wy - _origin_y) / _g_min;
    if (fi < 0.0 || fj < 0.0)
        return false;
    i = static_cast<unsigned int>(std::floor(fi));
    j = static_cast<unsigned int>(std::floor(fj));
    return (i < nx_ && j < ny_);
}

bool ConfigurationMapper::isPositionBlocked(unsigned int i, unsigned int j) const
{
    for (unsigned int k = 0; k < nz_; k++)
    {
        if (configuration_space[idx(i, j, k)] < 0.5f)
            return false;
    }
    return true;
}

double ConfigurationMapper::castCostRay3D(double start_x, double start_y,
                                           double dir_x, double dir_y,
                                           unsigned int theta_idx, double max_range) const
{
    // Normalize direction (NavRL-style: direction /= direction.norm())
    double len = std::sqrt(dir_x * dir_x + dir_y * dir_y);
    if (len < 1e-12) return max_range;
    double ndx = dir_x / len;
    double ndy = dir_y / len;

    // 3D c-space ray: traverse (x, y, θ) with θ as a third grid axis (3D DDA).
    // Endpoint unchanged: (x,y) direction is (ndx, ndy), and θ reaches θ_k at t = max_range.
    // Start at c-space origin (0,0,0): θ sweeps linearly from 0 to θ_k.
    double theta_k = static_cast<double>(theta_idx) * _a;
    double d_theta = (max_range > 1e-12) ? theta_k / max_range : 0.0; // rad per meter

    // Start cell: (start_x, start_y, θ = 0) → (i, j, 0)
    unsigned int i, j;
    if (!worldToGrid(start_x, start_y, i, j))
        return max_range; // out of map bounds → no hit
    int gi = static_cast<int>(i);
    int gj = static_cast<int>(j);
    int gk = 0;

    int sx = (ndx > 0.0) ? 1 : (ndx < 0.0 ? -1 : 0);
    int sy = (ndy > 0.0) ? 1 : (ndy < 0.0 ? -1 : 0);
    int sk = (d_theta > 0.0) ? 1 : 0;

    const double inf = std::numeric_limits<double>::infinity();

    // t to cross one cell along each axis
    double tdx = (ndx != 0.0)     ? std::fabs(_g_min / ndx) : inf;
    double tdy = (ndy != 0.0)     ? std::fabs(_g_min / ndy) : inf;
    double tdk = (d_theta > 0.0)  ? std::fabs(_a / d_theta) : inf; // = max_range / theta_idx

    // t to next cell boundary, starting at t = 0
    double tmx = inf, tmy = inf, tmk = inf;
    if (ndx > 0.0)      tmx = (_origin_x + (gi + 1) * _g_min - start_x) / ndx;
    else if (ndx < 0.0) tmx = (start_x - (_origin_x + gi * _g_min)) / (-ndx);
    if (ndy > 0.0)      tmy = (_origin_y + (gj + 1) * _g_min - start_y) / ndy;
    else if (ndy < 0.0) tmy = (start_y - (_origin_y + gj * _g_min)) / (-ndy);
    if (sk != 0)        tmk = _a / d_theta; // θ starts at 0, first boundary at θ = _a

    double t = 0.0;
    while (t <= max_range)
    {
        if (gi < 0 || gj < 0 || gk < 0 ||
            gi >= static_cast<int>(nx_) || gj >= static_cast<int>(ny_) || gk >= static_cast<int>(nz_))
            return max_range;

        if (costAt(static_cast<unsigned int>(gi),
                   static_cast<unsigned int>(gj),
                   static_cast<unsigned int>(gk)) > 0.5)
            return t; // first occupied voxel → (x,y) distance in meters

        if (tmx <= tmy && tmx <= tmk) { t = tmx; tmx += tdx; gi += sx; }
        else if (tmy <= tmx && tmy <= tmk) { t = tmy; tmy += tdy; gj += sy; }
        else { t = tmk; tmk += tdk; gk += sk; }
    }
    return max_range; // no collision
}

std::vector<float> ConfigurationMapper::computeCostRays3D(double center_x, double center_y,
                                                           double h_res, unsigned int num_beams,
                                                           double max_range) const
{
    // Output: beam-major layout, size = num_beams * nz_
    // For each horizontal beam direction, sweep all θ orientations.
    // This mirrors NavRL's 3D scan: H_beams × V_beams, where our "vertical"
    // dimension is the robot orientation θ.
    std::vector<float> cost_rays;
    cost_rays.reserve(static_cast<size_t>(num_beams) * nz_);

    for (unsigned int beam = 0; beam < num_beams; ++beam)
    {
        double angle = static_cast<double>(beam) * h_res;
        double dx = std::cos(angle);
        double dy = std::sin(angle);

        for (unsigned int k = 0; k < nz_; ++k)
        {
            double dist = castCostRay3D(center_x, center_y, dx, dy, k, max_range);
            cost_rays.push_back(static_cast<float>(dist));
        }
    }
    return cost_rays;
}

void ConfigurationMapper::collectCostRaySamples(
    double center_x, double center_y,
    double h_res, unsigned int num_beams, double max_range,
    unsigned int theta_idx, std::vector<CostRaySample> &samples) const
{
    samples.clear();

    for (unsigned int beam = 0; beam < num_beams; ++beam)
    {
        double angle = static_cast<double>(beam) * h_res;
        double dx = std::cos(angle);
        double dy = std::sin(angle);

        // Same 3D c-space traversal as castCostRay3D: θ sweeps linearly 0 → θ_k
        // over the ray, stepping (i, j, k) cells via 3D DDA.
        double theta_k = static_cast<double>(theta_idx) * _a;
        double d_theta = (max_range > 1e-12) ? theta_k / max_range : 0.0;

        unsigned int i, j;
        if (!worldToGrid(center_x, center_y, i, j))
            continue;
        int gi = static_cast<int>(i);
        int gj = static_cast<int>(j);
        int gk = 0;

        int sx = (dx > 0.0) ? 1 : (dx < 0.0 ? -1 : 0);
        int sy = (dy > 0.0) ? 1 : (dy < 0.0 ? -1 : 0);
        int sk = (d_theta > 0.0) ? 1 : 0;

        const double inf = std::numeric_limits<double>::infinity();
        double tdx = (dx != 0.0)     ? std::fabs(_g_min / dx) : inf;
        double tdy = (dy != 0.0)     ? std::fabs(_g_min / dy) : inf;
        double tdk = (d_theta > 0.0) ? std::fabs(_a / d_theta) : inf;

        double tmx = inf, tmy = inf, tmk = inf;
        if (dx > 0.0)      tmx = (_origin_x + (gi + 1) * _g_min - center_x) / dx;
        else if (dx < 0.0) tmx = (center_x - (_origin_x + gi * _g_min)) / (-dx);
        if (dy > 0.0)      tmy = (_origin_y + (gj + 1) * _g_min - center_y) / dy;
        else if (dy < 0.0) tmy = (center_y - (_origin_y + gj * _g_min)) / (-dy);
        if (sk != 0)       tmk = _a / d_theta;

        double t = 0.0;
        while (t <= max_range)
        {
            if (gi < 0 || gj < 0 || gk < 0 ||
                gi >= static_cast<int>(nx_) || gj >= static_cast<int>(ny_) || gk >= static_cast<int>(nz_))
                break;

            CostRaySample p;
            p.x    = static_cast<float>(center_x + t * dx);
            p.y    = static_cast<float>(center_y + t * dy);
            p.dist = static_cast<float>(t);
            p.beam = beam;
            samples.push_back(p);

            if (costAt(static_cast<unsigned int>(gi),
                       static_cast<unsigned int>(gj),
                       static_cast<unsigned int>(gk)) > 0.5)
                break; // first collision along the ray

            if (tmx <= tmy && tmx <= tmk) { t = tmx; tmx += tdx; gi += sx; }
            else if (tmy <= tmx && tmy <= tmk) { t = tmy; tmy += tdy; gj += sy; }
            else { t = tmk; tmk += tdk; gk += sk; }
        }
    }
}

void ConfigurationMapper::collectOccupiedVoxels(std::vector<CspaceVoxel> &voxels,
                                                  unsigned int crop_size, double z_scale,
                                                  double offset_x, double offset_y) const
{
    size_t total_cells = static_cast<size_t>(nx_) * ny_ * nz_;
    voxels.clear();
    voxels.reserve(total_cells / 10);

    unsigned int i_end = nx_ - crop_size;
    unsigned int j_end = ny_ - crop_size;

    for (unsigned int i = crop_size; i < i_end; ++i)
    {
        for (unsigned int j = crop_size; j < j_end; ++j)
        {
            for (unsigned int k = 0; k < nz_; ++k)
            {
                float cost = configuration_space[idx(i, j, k)];
                if (cost > 0.5f)
                {
                    CspaceVoxel v;
                    v.x = (_origin_x + static_cast<float>(i) * static_cast<float>(_g_min)) - static_cast<float>(offset_x);
                    v.y = (_origin_y + static_cast<float>(j) * static_cast<float>(_g_min)) - static_cast<float>(offset_y);
                    v.z = static_cast<float>(k) * static_cast<float>(z_scale);
                    v.cost = cost;
                    voxels.push_back(v);
                }
            }
        }
    }
}

namespace {

// Dot product of a theta slice with a reference vector
float theta_slice_dot_product(const std::vector<float>& cspace,
                               unsigned int ny, unsigned int nz,
                               unsigned int i, unsigned int j,
                               const std::vector<float>& ref)
{
    float product = 0.0f;
    size_t base = static_cast<size_t>(i) * ny * nz + static_cast<size_t>(j) * nz;
    for (unsigned int k = 0; k < nz; ++k)
    {
        product += cspace[base + k] * ref[k];
    }
    return product;
}

} // anonymous namespace

cv::Mat ConfigurationMapper::getXYProjectionImage(unsigned int scale) const
{
    if (configuration_space.empty()) {
        std::cerr << "Configuration space is empty!" << std::endl;
        return cv::Mat(); // empty
    }

    // Build reference patterns: red=first 1/3, green=middle 1/3, blue=last 1/3
    unsigned int third = nz_ / 3;
    std::vector<float> red_cfg(nz_, 0.0f);
    std::vector<float> green_cfg(nz_, 0.0f);
    std::vector<float> blue_cfg(nz_, 0.0f);
    for (unsigned int k = 0; k < third; ++k)
        red_cfg[k] = 1.0f;
    for (unsigned int k = third; k < 2 * third; ++k)
        green_cfg[k] = 1.0f;
    for (unsigned int k = 2 * third; k < nz_; ++k)
        blue_cfg[k] = 1.0f;

    // Create color image, white background
    cv::Mat img(static_cast<int>(nx_), static_cast<int>(ny_), CV_8UC3, cv::Scalar(255));

    for (unsigned int i = 0; i < nx_; ++i) {
        for (unsigned int j = 0; j < ny_; ++j) {
            float red_dist = theta_slice_dot_product(configuration_space, ny_, nz_, i, j, red_cfg);
            float green_dist = theta_slice_dot_product(configuration_space, ny_, nz_, i, j, green_cfg);
            float blue_dist = theta_slice_dot_product(configuration_space, ny_, nz_, i, j, blue_cfg);
            double average_dist = (red_dist + green_dist + blue_dist) / 3.0;
            red_dist -= static_cast<float>(average_dist);
            green_dist -= static_cast<float>(average_dist);
            blue_dist -= static_cast<float>(average_dist);
            double red = 1.0 / (1.0 + std::exp(-red_dist)) * 255;
            double green = 1.0 / (1.0 + std::exp(-green_dist)) * 255;
            double blue = 1.0 / (1.0 + std::exp(-blue_dist)) * 255;
            img.at<cv::Vec3b>(static_cast<int>(i), static_cast<int>(j)) =
                cv::Vec3b(static_cast<uchar>(blue), static_cast<uchar>(green), static_cast<uchar>(red));
        }
    }

    // Scale up for display
    int display_scale = scale > 0 ? static_cast<int>(scale) : 5;
    cv::Mat scaled;
    cv::resize(img, scaled, cv::Size(static_cast<int>(ny_) * display_scale,
                                     static_cast<int>(nx_) * display_scale),
               0, 0, cv::INTER_NEAREST);
    return scaled;
}

void ConfigurationMapper::visualizeXYProjection(const std::string &window_name, unsigned int scale)
{
    cv::Mat img = getXYProjectionImage(scale);
    if (img.empty())
        return;

    cv::imshow(window_name, img);
    cv::waitKey(1);
}

}