// cppimport
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <vector>
#include <queue>
#include <cmath>
#include <set>
#include <map>
#include <list>
#include <iostream>
#define INF 1000000000
namespace py = pybind11;

struct Node {
    int i;
    int j;
    float g;
    float h;
    float f;
    std::pair<int, int> parent;

    Node(int _i = INF, int _j = INF, float _g = INF, float _h = 0) : i(_i), j(_j), g(_g), h(_h), f(_g+_h){}
    bool operator<(const Node& other) const
    {
        return this->f < other.f or
               (this->f == other.f and (this->g < other.g or
                                       (this->g == other.g and (this->i < other.i or
                                                               (this->i == other.i and this->j < other.j)))));
    }
    bool operator>(const Node& other) const
    {
        return this->f > other.f or
               (this->f == other.f and (this->g > other.g or
                                       (this->g == other.g and (this->i > other.i or
                                                               (this->i == other.i and this->j > other.j)))));
    }
    bool operator==(const Node& other) const
    {
        return this->i == other.i and this->j == other.j;
    }
    bool operator==(const std::pair<int, int> &other) const
    {
        return this->i == other.first and this->j == other.second;
    }
};

struct FNode
{
    Node* node;
    float h2;
    FNode(Node* _node, float _h2):node(_node), h2(_h2){}
    bool operator<(const FNode& other) const
    {
        return this->h2 > other.h2 or (this->h2 == other.h2 and this->node->h < other.node->h);
    }
    bool operator>(const FNode& other) const
    {
        return this->h2 < other.h2 or (this->h2 == other.h2 and this->node->h > other.node->h);
    }
    bool operator==(const FNode& other) const
    {
        return *this->node == *other.node;
    }
};

class grid_planner
{
    std::pair<int, int> start;
    std::pair<int, int> goal;
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> OPEN;
    std::priority_queue<FNode, std::vector<FNode>, std::greater<FNode>> FOCAL;
    std::vector<std::vector<float>> grid;
    std::vector<std::vector<Node>> nodes;
    std::list<std::pair<int, int>> expanded_nodes;
	std::vector<std::vector<int>> expanded_flags;

    inline float h(std::pair<int, int> n)
    {
        int di = n.first - goal.first;
        int dj = n.second - goal.second;
        return std::min(di, dj)*std::sqrt(2.0) + std::abs(di - dj);
    }

    std::vector<std::pair<int,int>> get_neighbors(std::pair<int, int> node)
    {
        std::vector<std::pair<int,int>> neighbors;
        std::vector<std::pair<int,int>> deltas = {{0,1},{1,0},{-1,0},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1}};
        for(auto d:deltas)
        {
            std::pair<int,int> n(node.first + d.first, node.second + d.second);
            if(n.first >= 0 and n.first < int(grid.size()) and n.second >= 0 and n.second < int(grid.front().size()))
                neighbors.push_back(n);
        }
        return neighbors;
    }

    void compute_shortest_path()
    {
        Node current;
        while(!OPEN.empty() and !(current == goal))
        {
            current = OPEN.top();
            OPEN.pop();
            if(nodes[current.i][current.j].g < current.g)
                continue;
            expanded_nodes.push_back({current.i, current.j});
            for(auto n: get_neighbors({current.i, current.j})) {
                float cost = std::abs(grid[current.i][current.j] - grid[n.first][n.second]);
				if(current.i != n.first and current.j != n.second)
                    cost += std::sqrt(2.0);
                else
                    cost += 1.0;
                if(nodes[n.first][n.second].g > current.g + cost)
                {
                    OPEN.push(Node(n.first, n.second, current.g + cost, h(n)));
                    nodes[n.first][n.second].g = current.g + cost;
                    nodes[n.first][n.second].parent = {current.i, current.j};
                }
            }
        }
    }
    void reset(std::pair<int, int> s, std::pair<int, int> g)
    {
        start = s;
        goal = g;
        for(size_t i=0; i<grid.size(); i++)
            for(size_t j=0; j<grid.front().size(); j++)
                nodes[i][j] = Node(i, j, INF);
        nodes[start.first][start.second].g = 0;
        OPEN = std::priority_queue<Node, std::vector<Node>, std::greater<Node>>();
        OPEN.push(Node(start.first, start.second, 0, h(start)));
        FOCAL = std::priority_queue<FNode, std::vector<FNode>, std::greater<FNode>>();
        FOCAL.push(FNode(&nodes[start.first][start.second], 1));
        expanded_nodes.clear();
    }
public:
    std::list<std::pair<int, int>> find_focal_path(std::pair<int, int> s, std::pair<int, int> g, std::vector<std::vector<float>> heatmap)
    {
        reset(s, g);
        Node current;
		expanded_flags = std::vector<std::vector<int>>(heatmap.size(), std::vector<int>(heatmap.front().size(), 0));
        while(!FOCAL.empty() and !(current == goal))
        {
            current = *FOCAL.top().node;
            FOCAL.pop();
            if(expanded_flags[current.i][current.j] == 1)
                continue;
			expanded_flags[current.i][current.j] = 1;
            expanded_nodes.push_back({current.i, current.j});
            for(auto n: get_neighbors({current.i, current.j})) {
                float cost = std::abs(grid[current.i][current.j] - grid[n.first][n.second]);
                if(current.i != n.first and current.j != n.second)
                    cost += std::sqrt(2.0);
                else
                    cost += 1.0;
                if(expanded_flags[n.first][n.second] == 0)
                {
                    nodes[n.first][n.second].g = current.g + cost;
                    nodes[n.first][n.second].h = h(n);
                    nodes[n.first][n.second].f = nodes[n.first][n.second].g + nodes[n.first][n.second].h;
                    nodes[n.first][n.second].parent = {current.i, current.j};
                    FOCAL.push(FNode(&nodes[n.first][n.second], heatmap[n.first][n.second]));
                }
            }
        }
        return get_path();
    }
	
	std::list<std::pair<int, int>> find_focal_path_reexpand(std::pair<int, int> s, std::pair<int, int> g, std::vector<std::vector<float>> heatmap)
    {
        reset(s, g);
        Node current;
        while(!FOCAL.empty() and !(current == goal))
        {
            current = *FOCAL.top().node;
            FOCAL.pop();
            if(nodes[current.i][current.j].g < current.g )
                continue;
            expanded_nodes.push_back({current.i, current.j});
            for(auto n: get_neighbors({current.i, current.j})) {
                float cost = std::abs(grid[current.i][current.j] - grid[n.first][n.second]);
                if(current.i != n.first and current.j != n.second)
                    cost += std::sqrt(2.0);
                else
                    cost += 1.0;
                if(nodes[n.first][n.second].g > current.g + cost)
                {
                    nodes[n.first][n.second].g = current.g + cost;
                    nodes[n.first][n.second].h = h(n);
                    nodes[n.first][n.second].f = nodes[n.first][n.second].g + nodes[n.first][n.second].h;
                    nodes[n.first][n.second].parent = {current.i, current.j};
                    FOCAL.push(FNode(&nodes[n.first][n.second], heatmap[n.first][n.second]));
                }
            }
        }
        return get_path();
    }
	
    int get_num_expansions()
    {
        return expanded_nodes.size();
    }

    float get_path_cost()
    {
        return nodes[goal.first][goal.second].g;
    }

    grid_planner(std::vector<std::vector<float>> _grid):grid(_grid)
    {
        nodes = std::vector<std::vector<Node>>(grid.size(), std::vector<Node>(grid.front().size(), Node()));
    }

    std::list<std::pair<int, int>> find_path(std::pair<int, int> s, std::pair<int, int> g)
    {
        reset(s, g);
        compute_shortest_path();
        return get_path();
    }

    std::list<std::pair<int, int>> get_path()
    {
        std::list<std::pair<int, int>> path;
        std::pair<int, int> next_node(INF,INF);
        if(nodes[goal.first][goal.second].g < INF)
            next_node = goal;
        if(next_node.first < INF and (next_node.first != start.first or next_node.second != start.second))
        {
            while (nodes[next_node.first][next_node.second].parent != start) {
                path.push_back(next_node);
                next_node = nodes[next_node.first][next_node.second].parent;
            }
            path.push_back(next_node);
            path.push_back(start);
            path.reverse();
        }
        return path;
    }

    std::vector<std::vector<float>> find_heatmap(std::pair<int, int> s, std::pair<int, int> g)
    {
        reset(s, {-1,-1});
        compute_shortest_path();
        std::vector<std::vector<float>> heatmap(grid.size(), std::vector<float>(grid.size(), 0));
        for(size_t i = 0; i < grid.size(); i++)
            for(size_t j = 0; j < grid.size(); j++)
                heatmap[i][j] = nodes[i][j].g;
        reset(g, {-1,-1});
        compute_shortest_path();
        for(size_t i = 0; i < grid.size(); i++)
            for(size_t j = 0; j < grid.size(); j++)
                heatmap[i][j] += nodes[i][j].g;

		reset(s,g);
        compute_shortest_path();
		auto path = get_path();
		reset(path.front(), {-1,-1});
        OPEN = std::priority_queue<Node, std::vector<Node>, std::greater<Node>>();
		for(auto it = path.begin(); it != path.end(); it++)
		{
			nodes[it->first][it->second].g = 0;
			OPEN.push(Node(it->first, it->second, 0, 0));
		}
        compute_shortest_path();
        float min_g = INF;
		for(size_t i = 0; i < grid.size(); i++)
            for(size_t j = 0; j < grid.size(); j++)
            {
                heatmap[i][j] += nodes[i][j].g;
                min_g = std::fmin(min_g, heatmap[i][j]);
            }
        for(size_t i = 0; i < grid.size(); i++)
            for(size_t j = 0; j < grid.size(); j++)
                heatmap[i][j] = min_g/heatmap[i][j];
        return heatmap;
    }

    std::vector<std::vector<int>> get_expansions()
    {
        std::vector<std::vector<int>> expansions(grid.size(), std::vector<int>(grid.size(), 0));
        for(size_t i = 0; i < grid.size(); i++)
            for(size_t j = 0; j < grid.size(); j++)
                if(nodes[i][j].g < INF)
                    expansions[i][j] = 1;
        for(auto n: expanded_nodes)
            expansions[n.first][n.second] = 2;
        return expansions;
    }
};

PYBIND11_MODULE(grid_planner, m) {
    py::class_<grid_planner>(m, "grid_planner")
            .def(py::init<std::vector<std::vector<float>>>())
            .def("find_path", &grid_planner::find_path)
            .def("get_path", &grid_planner::get_path)
            .def("find_heatmap", &grid_planner::find_heatmap)
            .def("get_expansions", &grid_planner::get_expansions)
            .def("get_num_expansions", &grid_planner::get_num_expansions)
            .def("get_path_cost", &grid_planner::get_path_cost)
            .def("find_focal_path", &grid_planner::find_focal_path)
            .def("find_focal_path_reexpand", &grid_planner::find_focal_path_reexpand);
}

/*
<%
setup_pybind11(cfg)
%>
*/