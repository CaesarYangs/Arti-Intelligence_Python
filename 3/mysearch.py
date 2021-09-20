import collections
import sys
from collections import deque
from turtle import distance
import operator


class Problem:  #这是抽象类 不需要在这一类写方法函数 以后的应用类都会重载
    """The abstract class for a formal problem. You should subclass
    this and implement the methods actions and result, and possibly
    __init__, goal_test, and path_cost. Then you will create instances
    of your subclass and solve them with the various search functions."""

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal. Your subclass's constructor can add
        other arguments."""
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        raise NotImplementedError

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        raise NotImplementedError



    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough."""
        # if isinstance(self.goal, list):
        #     return is_in(state, self.goal)
        # else:
        #     return state == self.goal
        return state == self.goal


    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2. If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def value(self, state):
        """For optimization problems, each state has a value. Hill Climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError


# ______________________________________________________________________________


class Node:
    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state. Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node. Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):   #构造函数
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        """[Figure 3.10]"""
        next_state = problem.result(self.state, action)
        next_node = Node(next_state, self, action, problem.path_cost(self.path_cost, self.state, action, next_state))
        return next_node

    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        return [node.action for node in self.path()[1:]]

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    def path_count_re(self):
        return self.path_cost

    # We want for a queue of nodes in breadth_first_graph_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        # We use the hash value of the state
        # stored in the node instead of the node
        # object itself to quickly search a node
        # with the same state in a Hash Table
        return hash(self.state)








#############################
neighbor_map = {'Arad':['Zerind','Sibiu','Timisoara'],'Bucharest':['Urziceni', 'Pitesti', 'Giurgiu', 'Fagaras'],'Craiova':['Drobeta', 'Rimnicu', 'Pitesti'],'Drobeta':['Mehadia'],'Eforie':['Hirsova'],
'Fagaras':['Sibiu'],'Hirsova':['Urziceni'],'Iasi':['Vaslui','Neamt'],'Lugoj':['Timisoara','Mehadia'],
'Oradea':['Zerind', 'Sibiu'],'Pitesti':['Rimnicu'],'Rimnicu':['Sibiu'],'Urziceni':['Vaslui']}
               
neighbormapWithweight = {'Arad':{'Zerind':75,'Sibiu':140,'Timisoara':118},
                         'Bucharest':{'Urziceni':85, 'Pitesti':101, 'Giurgiu':90,'Fagaras':211},
                         'Craiova':{'Drobeta':120, 'Rimnicu':146, 'Pitesti':138},
                         'Drobeta':{'Mehadia':75},
                         'Eforie':{'Hirsova':86},
                         'Fagaras':{'Bucharest':211},
                         'Hirsova':{'Urziceni':98},
                         'Iasi':{'Vaslui':92,'Neamt':87},
                         'Lugoj':{'Mehadia':70},
                         'Oradea':{'Sibiu':151},
                         'Pitesti':{'Bucharest':101},
                         'Rimnicu':{'Pitesti':97,'Craiova':146},
                         'Urziceni':{'Vaslui':142},
                         'Sibiu':{'Fagaras':99,'Rimnicu':80},
                         'Zerind':{'Oradea':71},
                         'Timisoara':{'Lugoj':111}
                         }
                         
                         
                         
romania_map_locations = dict(
    Arad=(91, 492), Bucharest=(400, 327), Craiova=(253, 288),
    Drobeta=(165, 29), Eforie=(562, 293), Fagaras=(305, 449),
    Giurgiu=(375, 270), Hirsova=(534, 350), Iasi=(473, 506),
    Lugoj=(165, 379), Mehadia=(168, 339), Neamt=(406, 537),
    Oradea=(131, 571), Pitesti=(320, 368), Rimnicu=(233, 410),
    Sibiu=(207, 457), Timisoara=(94, 410), Urziceni=(456, 350),
    Vaslui=(509, 444), Zerind=(108, 531))

distance_to_Bu = dict({'Arad':366,'Sibiu':253,'Rimnicu':193,'Fagaras':178,'Pitesti':98,'Bucharest':0,'Timisoara':329,'Zerind':374,'Oradea':380,'Craiova':160,'Urziceni':80,'Giurgiu':77,'Lugoj':244,'Mehadia':241})
    #各个城市坐标信息

class GraphProblem(Problem): #实例化Problem抽象类
    def __init__(self, initial, goal, graph):
        Problem.__init__(self, initial, goal)
        self.graph = graph

    def successor(self, A):
        "Return a list of (action, result) pairs."
        return [(B, B) for B in self.graph.get(A).keys()]

    def path_cost(self, cost_so_far, A, action, B):
        return cost_so_far + (self.graph.get(A, B) or 10000000)

    def heuristic_star(self, node):
        "h function is straight-line distance from a node's state to goal."
        locs = getattr(self.graph, 'locations', None)
        if locs:
            return int(distance(locs[node.state], locs[self.goal]))
        else:
            return 10000000
    def heuristic_greedy(self, node):
        return 0


class StackFrontier():
    def __init__(self):
        self.frontier = {}

    def add(self, node):
        self.frontier[node.state] = node.path_cost

    def contains_state(self, state):
        return any(node.state == state for node in self.frontier)

    def empty(self):
        return len(self.frontier) == 0

    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        else:
            return self.frontier.popitem()


class QueueFrontier(StackFrontier,Node):
    def __init__(self):
        self.frontier = collections.OrderedDict()
    # def quene_sort(self):
    #     # sorted(self.frontier.keys())
    #     sorted(self.frontier.items(), key=operator.itemgetter(1))

    def add(self, node):
        self.frontier[node.state] = node.path_cost
        self.frontier = collections.OrderedDict(sorted(self.frontier.items(), key=lambda t: t[1]))  #利用有序字典建立优先级字典队列 每次插入新的待探索元素后自动排序

        # self.frontier.update({node.state,node.path_cost})

    def get(self):
        if self.empty():
            raise Exception("empty frontier")
        else:
            # self.frontier.popitem()
            return next(iter(self.frontier.items()))

    def getDepth(self):
        return self.depth

    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        else:
            # self.frontier.popitem()
            re = next(iter(self.frontier.items()))  # get the first item
            self.frontier.pop(next(iter(self.frontier)))
            return re



#---

#---c

class Romania_trip(Problem):
    def __init__(self,initial,goal,graph,graph_weight,distance_to):
        Problem.__init__(self, initial, goal)
        self.graph = graph
        self.goal = goal
        self.state = Node
        self.graphWeight = graph_weight
        self.distance = distance_to
        # print("init succeed")
        # print("initial = {}   goal = {}".format(self.initial,self.goal))
        # print(graph[initial])


    def goal_test(self, state):
        # if isinstance(self.goal, list):
        #     return is_in(state, self.goal)
        # else:
        #     return state == self.goal
        return state != self.goal


    def greedy_best_first_graph_search(self):
        print('greedy_best_first_graph_search:')
        mainpath = 0
        start = Node(state=self.initial,path_cost=0)
        frontier = StackFrontier()
        frontier.add(start)
        self.state = self.initial

        while self.goal_test(self.state[0]):
            self.state = frontier.remove()
            if self.goal_test(self.state[0])==0:
                break;
            print(self.state,end="-->")
            it = self.graphWeight[self.state[0]]
            min_num = 100000
            min_name = ''
            for i in it.keys():
                # frontier.add(Node(state=i,path_cost=it[i]))
                if(it[i]<min_num):
                    min_num = it[i]
                    min_name = i
            mainpath = mainpath + min_num
            frontier.add(Node(state=min_name,path_cost=min_num))


        print(self.goal,end='\n')
        print(mainpath)

    def astar_search(self, h=None):
        path = []
        path.append(0)
        pre_path = 0
        start = Node(state=self.initial, path_cost=0)
        frontier = QueueFrontier()
        frontier.add(start)
        self.state = frontier.get()
        while self.goal_test(self.state[0]):
            pre = self.graphWeight[self.state[0]]
            self.state = frontier.remove()
            now = self.state[0]
            # print(self.state[0])

            print(self.state[0], end=' ')
            if(self.state[0]!='Arad'):
                pre_path = pre_path + pre[now]
                print(pre_path,end='-->')
                path.append(pre_path)
            else:
                print(pre_path,end='-->')

            it = self.graphWeight[self.state[0]]  # 取得此时最小元素的子节点
            for i in it:  # 循环遍历次数组 加入优先级队列中待命
                frontier.add(Node(state=i, path_cost=it[i] + self.distance[i]))
            # print(frontier.frontier)
        print('end!')
        # for i in path:
        #     print(i,'-->',end='')

# rt = Romania_trip('Arad','Bucharest',neighbor_map,neighbormapWithweight,distance_to_Bu)
# # rt.greedy_best_first_graph_search()
# rt.astar_search()




################################################
class EightPuzzle(Problem):
    """ The problem of sliding tiles numbered from 1 to 8 on a 3x3 board, where one of the
    squares is a blank. A state is represented as a tuple of length 9, where  element at
    index i represents the tile number  at index i (0 if it's an empty square) """

    def __init__(self, initial, goal=(1, 2, 3, 4, 5, 6, 7, 8, 0)):
        """ Define goal state and initialize a problem """
        self.state = initial
        super().__init__(initial, goal)


    def find_blank_square(self, state):
        """Return the index of the blank square in a given state"""

        return state.index(0)

    def actions(self, state):
        """ Return the actions that can be executed in the given state.
        The result would be a list, since there are only four possible actions
        in any given state of the environment """

        possible_actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        index_blank_square = self.find_blank_square(state)

        if index_blank_square % 3 == 0:
            possible_actions.remove('LEFT')
        if index_blank_square < 3:
            possible_actions.remove('UP')
        if index_blank_square % 3 == 2:
            possible_actions.remove('RIGHT')
        if index_blank_square > 5:
            possible_actions.remove('DOWN')

        return possible_actions

    def result(self, state, action):
        """ Given state and action, return a new state that is the result of the action.
        Action is assumed to be a valid action in the state """

        # blank is the index of the blank square
        blank = self.find_blank_square(state)
        new_state = list(state)

        delta = {'UP': -3, 'DOWN': 3, 'LEFT': -1, 'RIGHT': 1}
        neighbor = blank + delta[action]
        new_state[blank], new_state[neighbor] = new_state[neighbor], new_state[blank]

        return tuple(new_state)

    def goal_test(self, state):
        """ Given a state, return True if state is a goal state or False, otherwise """

        return state != self.goal

    def check_solvability(self, state):
        """ Checks if the given state is solvable """

        inversion = 0
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                if (state[i] > state[j]) and state[i] != 0 and state[j] != 0:
                    inversion += 1

        return inversion % 2 == 0

    def show_state(self,state):
        for i in range(len(state)):
            print(state[i], '|', end='')
            if (i+1)%3 == 0:
                print()

        return

    def h(self, state):
        """ Return the heuristic value for a given state. Default heuristic function used is 
        h(n) = number of misplaced tiles """

        ''' please implement your heuristic function'''
        m_distance = 0
        pos = state
        pos_goal = self.goal

        for i in range(9):
            if pos[i]==0:
                continue
            x = (pos_goal.index(pos[i]))// 3
            y = (pos_goal.index(pos[i])) % 3
            x_o = (i) // 3
            y_o = (i) % 3
            m_distance = m_distance + abs(x-x_o) + abs(y-y_o)

        return m_distance

    def astar_solve(self):
        possible_mid = []
        start = Node(state=self.initial, path_cost=0)
        frontier = QueueFrontier()
        frontier.add(start)
        self.state = frontier.get()
        depth = 0
        num = 0

        print('---start---')
        while self.goal_test(self.state):
            self.state = frontier.remove()[0]

            print('--state:')
            self.show_state(self.state)

            possible_mid = self.actions(self.state)
            for i in range(len(possible_mid)):
                now_state = self.result(self.state, possible_mid[i])
                step = depth + self.h(now_state)
                frontier.add(Node(state=now_state,path_cost=step))
            depth = depth + 1

        print()
        print(depth)

        # path_depth = 0
        # self.show_state(self.state)
        # print()
        # self.show_state(self.goal)
        # print('---start---')
        # # print(self.check_solvability(self.state))
        # # print(self.actions(self.state))
        # while self.goal_test(self.state):
        #     possible_mid = self.actions(self.state)
        #     min_step = 10000
        #     for i in range(len(possible_mid)):
        #         now_state = self.result(self.state,possible_mid[i])
        #         step = path_depth + self.h(now_state)
        #         if step<min_step:
        #             min_step = step
        #             min_state = now_state
        #     self.state = min_state
        #     print('--state:')
        #     self.show_state(self.state)
        # path_depth = path_depth+1
        # print()
        # print('m:',self.h(self.state))


# test = (2,8,3,1,6,4,7,0,5)
# test_goal = (1,2,3,8,0,4,7,6,5)
# pz_test = EightPuzzle(initial=test,goal=test_goal)
# pz_test.astar_solve()

eight_initialu = (7,2,4,5,0,6,8,3,1)
goal=(1,2,3,4,5,6,7,8,0)
eight_initial = (2,4,3,1,5,6,7,8,0)
pz = EightPuzzle(initial=eight_initial)
pz.astar_solve()
