#include "nav_rvo.h"

#include <ros/ros.h>
#include <utility>

namespace RVO
{

RVOPlanner::RVOPlanner(std::string simulation) : simulator(std::move(simulation))
{
    sim = new RVO::RVOSimulator();
};

RVOPlanner::~RVOPlanner()
{
    for (auto *velocity : newVelocities)
        delete velocity;
    delete sim;
}

void RVOPlanner::setupScenario(float neighborDist, size_t maxNeighbors, float timeHorizon, float timeHorizonObst, float radius, float maxSpeed)
{
    neighbor_dist_ = neighborDist;
    max_neighbors_ = maxNeighbors;
    time_horizon_ = timeHorizon;
    time_horizon_obst_ = timeHorizonObst;
    radius_ = radius;
    max_speed_ = maxSpeed;
    sim->setAgentDefaults(neighborDist, maxNeighbors, timeHorizon, timeHorizonObst, radius, maxSpeed);
}

// set the goal manually
// default: invert direction
// random: change the goal every step

void RVOPlanner::setGoal()
{
    for (size_t i = 0; i < sim->getNumAgents(); ++i)
    {
        if (absSq(goals[i] - sim->getAgentPosition(i)) < goal_threshold)
        {
            // goals[i] = -sim->getAgentPosition(i);
            goals[i] = -goals[i];
        }
    }
}

void RVOPlanner::setGoal(std::vector<geometry_msgs::Point> set_goals)
{
    goals.clear();
    int num_agent = sim->agents_.size();
    if (set_goals.size() < num_agent)
        std::cout << "error:The num of goals is less than agents" << std::endl;
    else
    {
        for (int i = 0; i < num_agent; i++)
        {
            float x = set_goals[i].x;
            float y = set_goals[i].y;

            goals.emplace_back(Vector2(x, y));
            std::cout<<"goal"+ std::to_string(i+1) + ":["<<x<<","<<y<<"]"<<std::endl;
        }
    }
}


void RVOPlanner::randomOnceGoal(const float limit_goal[4])
{
    float x_min = limit_goal[0];
    float x_max = limit_goal[1];
    float y_min = limit_goal[2];
    float y_max = limit_goal[3];

    std::random_device rd;
    std::default_random_engine e(rd());
    std::uniform_real_distribution<float> ux(x_min, x_max);
    std::uniform_real_distribution<float> uy(y_min, y_max);
    std::uniform_int_distribution<int> ur(0, 10);

    goals.resize(sim->getNumAgents());
    for (size_t i = 0; i < sim->getNumAgents(); ++i)
    {

        float x = ux(e);
        float y = uy(e);
        int rand = ur(e);
    
        goals[i] = (Vector2(x, y));

        std::cout<< "random once successfully" <<std::endl;
    }
       
}

void RVOPlanner::randGoal(const float limit_goal[4], const std::string &model)
{

    float x_min = limit_goal[0];
    float x_max = limit_goal[1];
    float y_min = limit_goal[2];
    float y_max = limit_goal[3];

    std::random_device rd;
    std::default_random_engine e(rd());
    std::uniform_real_distribution<float> ux(x_min, x_max);
    std::uniform_real_distribution<float> uy(y_min, y_max);
    std::uniform_int_distribution<int> ur(0, 10);

    for (size_t i = 0; i < sim->getNumAgents(); ++i)
    {

        float x = ux(e);
        float y = uy(e);
        int rand = ur(e);

        if (!IfInitial)
            goals.emplace_back(x, y);

        else if (model == "default")
        {
            if (absSq(goals[i] - sim->getAgentPosition(i)) < goal_threshold)
                goals[i] = (Vector2(x, y));
        }
        else if (model == "random")
        {
            if (rand > 8)
                goals[i] = (Vector2(x, y));
        }
    }
}

bool RVOPlanner::arrived()
{
    if (goals.size() < sim->getNumAgents())
        return false;

    bool reach = true;

    for (size_t i = 0; i < sim->getNumAgents(); ++i)
    {
        if (absSq(goals[i] - sim->getAgentPosition(i)) >= goal_threshold)
            reach = false;
    }

    return reach;
}



void RVOPlanner::setInitial()
{
    IfInitial = (!goals.empty()) && (!sim->agents_.empty());
}

void RVOPlanner::setPreferredVelocities()
{
    for (size_t i = 0; i < sim->getNumAgents(); ++i)
    {
        if (i >= goals.size())
        {
            sim->setAgentPrefVelocity(i, RVO::Vector2(0.0f, 0.0f));
            continue;
        }

        if (absSq(goals[i] - sim->getAgentPosition(i)) < goal_threshold)
        {
            // Agent is within one radius of its goal, set preferred velocity to zero
            sim->setAgentPrefVelocity(i, RVO::Vector2(0.0f, 0.0f));
        }
        else
        {
            sim->setAgentPrefVelocity(i, normalize(goals[i] - sim->getAgentPosition(i)));
        }
    }
}

void RVOPlanner::updateState_gazebo(gazebo_msgs::ModelStates::ConstPtr model_msg, std::string agent_name)
{
    if (simulator == "gazebo")
    {
        std::vector<size_t> agent_indices;
        std::vector<std::string> current_agent_names;

        // Filter the actual Gazebo model list instead of assuming names are
        // prefix+0, prefix+1, ... up to the total number of world models.
        // Gazebo may publish several intermediate ModelStates messages while
        // models are being spawned, so the matching set can change at startup.
        for (size_t i = 0; i < model_msg->name.size(); ++i)
        {
            const std::string &model_name = model_msg->name[i];
            if (model_name.compare(0, agent_name.size(), agent_name) == 0 &&
                i < model_msg->pose.size() && i < model_msg->twist.size())
            {
                agent_indices.push_back(i);
                current_agent_names.push_back(model_name);
            }
        }

        if (current_agent_names != agent_names)
        {
            // Recreate the complete simulator so its kd-tree cannot retain
            // pointers to agents from a previous, incomplete spawn set.
            for (auto *velocity : newVelocities)
                delete velocity;
            newVelocities.clear();
            delete sim;
            sim = new RVO::RVOSimulator();
            sim->setAgentDefaults(neighbor_dist_, max_neighbors_, time_horizon_,
                                  time_horizon_obst_, radius_, max_speed_);
            goals.clear();
            IfInitial = false;
            agent_names = current_agent_names;

            ROS_INFO("RVO agent set changed; rebuilding with %zu agents", agent_names.size());

            for (size_t index : agent_indices)
            {
                sim->addAgent(RVO::Vector2(model_msg->pose[index].position.x,
                                           model_msg->pose[index].position.y));
                sim->agents_.back()->velocity_ =
                    RVO::Vector2(model_msg->twist[index].linear.x,
                                 model_msg->twist[index].linear.y);
            }
        }
        else
        {
            for (size_t count = 0; count < agent_indices.size(); ++count)
            {
                const size_t index = agent_indices[count];
                sim->agents_[count]->position_ =
                    RVO::Vector2(model_msg->pose[index].position.x,
                                 model_msg->pose[index].position.y);
                sim->agents_[count]->velocity_ =
                    RVO::Vector2(model_msg->twist[index].linear.x,
                                 model_msg->twist[index].linear.y);
            }
        }
    }
    else
        std::cout << "error: please check the simulator" << std::endl;
}

std::vector<RVO::Vector2 *> RVOPlanner::step()
{
    sim->kdTree_->buildAgentTree();
    for (auto *velocity : newVelocities)
        delete velocity;
    newVelocities.clear();

    for (auto & agent : sim->agents_)
    {
        agent->computeNeighbors();
        agent->computeNewVelocity();
        auto *new_vel = new Vector2(agent->newVelocity_.x(), agent->newVelocity_.y());
        newVelocities.push_back(new_vel);
    }

    return newVelocities;
}

}; // namespace RVO
