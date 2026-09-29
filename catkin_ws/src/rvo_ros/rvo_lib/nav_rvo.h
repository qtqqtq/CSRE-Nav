#ifndef NAV_ROV_H_
#define NAV_ROV_H_

#include "Agent.h"
#include "KdTree.h"
#include "Definitions.h"
#include "Obstacle.h"
#include "gazebo_msgs/ModelStates.h"
#include "geometry_msgs/Point.h"
#include "RVOSimulator.h"
#include <string>
#include <random>
#include <vector>

namespace RVO {

    class Agent;
    class Obstacle;
    class KdTree;

    class RVOPlanner{
    public:
        RVOPlanner(std::string simulation);
        ~RVOPlanner();

        void setupScenario(float neighborDist, size_t maxNeighbors, float timeHorizon, float timeHorizonObst, float radius, float maxSpeed);

        void updateState_gazebo(gazebo_msgs::ModelStates::ConstPtr model_msg, std::string agent_name);

        void setGoal();
        void randGoal(const float limit_goal[4], const std::string &model="default");
        void randomOnceGoal(const float limit_goal[4]);
        bool arrived();
        void setGoal(std::vector<geometry_msgs::Point> set_goals);
        void setInitial();
        void setPreferredVelocities();

        std::vector<RVO::Vector2*>  step();
        float goal_threshold = 0.03;
        
        
        
    private:

        RVO::RVOSimulator* sim;
        std::string simulator;
        std::vector <RVO::Vector2> goals;
        bool IfInitial = false;
        std::vector<RVO::Vector2 *> newVelocities;
        std::vector<std::string> agent_names;
        float neighbor_dist_ = 4.0f;
        size_t max_neighbors_ = 10;
        float time_horizon_ = 10.0f;
        float time_horizon_obst_ = 5.0f;
        float radius_ = 0.3f;
        float max_speed_ = 0.2f;
        

        friend class Agent;
        friend class KdTree;
        friend class Obstacle;
    };
}

#endif
