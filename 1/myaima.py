#!/usr/bin/env python
# coding: utf-8

# In[3]:


#import sys
#sys.path.insert(0,'/home1/user/lr/aima-python-master')


# In[4]:


from agents import *
#from notebook import psource


# In[14]:


#psource(TrivialVacuumEnvironment)


# In[15]:


# These are the two locations for the two-state environment
loc_A, loc_B = (0, 0), (1, 0)

# Initialize the two-state environment
trivial_vacuum_env = TrivialVacuumEnvironment()

# Check the initial state of the environment
print("State of the Environment: {}.".format(trivial_vacuum_env.status))


# In[16]:


loc_A = (0, 0)
loc_B = (1, 0)



"""We change the simpleReflexAgentProgram so that it doesn't make use of the Rule class"""
def SimpleReflexAgentProgram():
    """This agent takes action based solely on the percept. [Figure 2.10]"""


    def program(percept):
        loc, status = percept
        if status=='Dirty':
            print('--action:Suck')
            action = 'Suck'
        elif loc==(0, 0):
            print('--action:turn right')
            action = 'Right'
        else:
            print('--action:turn left')
            action='Left'
        return action


    return program

        
# Create a simple reflex agent the two-state environment
program = SimpleReflexAgentProgram()
simple_reflex_agent = Agent(program)


# In[17]:


trivial_vacuum_env.add_thing(simple_reflex_agent)

print("SimpleReflexVacuumAgent is located at {}.".format(simple_reflex_agent.location))


# In[18]:


# Run the environment

print('-------------------Simple Reflex Agent----------------------')
for i in range(3):
    trivial_vacuum_env.step()
    print("State of the Environment: {}.".format(trivial_vacuum_env.status))
    print("SimpleReflexVacuumAgent is located at {}.".format(simple_reflex_agent.location))
    print('step',i+1,'finish')
    print()
print('------------------------------------------------------------')

# trivial_vacuum_env.step()
#
# # Check the current state of the environment
# print("State of the Environment: {}.".format(trivial_vacuum_env.status))
#
# print("SimpleReflexVacuumAgent is located at {}.".format(simple_reflex_agent.location))




# In[24]:


trivial_vacuum_env.delete_thing(simple_reflex_agent)


# In[22]:
####################################

table = {((loc_A, 'Clean'),): 'Right',
             ((loc_A, 'Dirty'),): 'Suck',
             ((loc_B, 'Clean'),): 'Left',
             ((loc_B, 'Dirty'),): 'Suck',
             ((loc_A, 'Dirty'), (loc_A, 'Clean')): 'Right',
             ((loc_A, 'Clean'), (loc_B, 'Dirty')): 'Suck',
             ((loc_B, 'Clean'), (loc_A, 'Dirty')): 'Suck',
             ((loc_B, 'Dirty'), (loc_B, 'Clean')): 'Left',
             ((loc_A, 'Dirty'), (loc_A, 'Clean'), (loc_B, 'Dirty')): 'Suck',
             ((loc_B, 'Dirty'), (loc_B, 'Clean'), (loc_A, 'Dirty')): 'Suck'
        }


# In[23]:

state = 0
# Create a table-driven agent
table_driven_agent = Agent(program=TableDrivenAgentProgram(table=table))

def ReflexAgentwithStateProgram():

    model = {loc_A:None,loc_B:None}
    def program(percept):

        loc, status = percept
        model[loc] = status

        if model[loc_A] == model[loc_B]=='Clean':
            action = 'NoOp'
            print('--=action:Nothing to do   NoOp')
        elif status == 'Dirty':
            action = 'Suck'
            print('--=action:Suck')
        elif loc_A == loc_A:
            action = 'Right'
            print('--=action:turn right')
        else:
            action = 'Left'
            print('--=action:turn left')

        return action

        # global state
        # if status=='Dirty':
        #     print('--=action:Suck')
        #     action = 'Suck'
        #     state+=1
        # elif loc==(0, 0):
        #     if(status=='Clean'):
        #         state+=1
        #     print('--=action:turn right')
        #     action = 'Right'
        # else:
        #     if (status == 'Clean'):
        #         state+=1
        #     print('--=action:turn left')
        #     action='Left'
        #
        # return action



    return program

# In[25]:

program = ReflexAgentwithStateProgram()
table_driven_agent = Agent(program)

# Add the table-driven agent to the environment
trivial_vacuum_env.add_thing(table_driven_agent)

print("TableDrivenVacuumAgent is located at {}.".format(table_driven_agent.location))


# In[26]:


# Run the environment
print('------------------------with State--------------------------')
for i in range(3):
    if(state>=2):
        break
    trivial_vacuum_env.step()
    print("State of the Environment: {}.".format(trivial_vacuum_env.status))
    print("TableDrivenVacuumAgent is located at {}.".format(table_driven_agent.location))
    print('step', i + 1, 'finish')
    print()
print('------------------------------------------------------------')


# trivial_vacuum_env.step()
#
# # Check the current state of the environment
# print("State of the Environment: {}.".format(trivial_vacuum_env.status))
#
# print("TableDrivenVacuumAgent is located at {}.".format(table_driven_agent.location))


# In[ ]:




