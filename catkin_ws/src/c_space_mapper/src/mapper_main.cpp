#include <ros/ros.h>
#include "c_space_mapper/mapper_node.hpp"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "c_space_mapper_node");
    c_space_mapper::MapperNode node;
    ros::spin();
    return 0;
}
