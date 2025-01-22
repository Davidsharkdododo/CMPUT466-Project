import numpy as np

class GreedyAgent:
    def __init__(self):
        """
        Initialize the GreedyAgent with default parameters.
        """
        self.num_actions = 5  # Default number of actions (arms)
        self.q_values = [0, 0, 0, 0, 0]  # Initialize Q-values to zero
        self.arm_count = [0, 0, 0, 0, 0]  # Initialize action counts to zero
        self.last_action = None  # Initialize the last action to None

    def agent_step(self, reward, observation=None):
        """
        Takes one step for the agent. It takes in a reward and observation and 
        returns the action the agent chooses at that time step.
        
        Arguments:
        reward -- float, the reward the agent received from the environment after taking the last action.
        observation -- Do not worry about this, as we will not use it until future lessons
        Returns:
        current_action -- int, the action chosen by the agent at the current time step.
        """
        if self.last_action is not None:
            # Update Q values
            action = self.last_action
            self.arm_count[action] += 1  # Increment the counter for the previous action
            step_size = 1 / self.arm_count[action]  # Calculate step size
            self.q_values[action] += step_size * (reward - self.q_values[action])  # Update Q-value for the previous action

        # Select the next action using argmax
        current_action = np.argmax(self.q_values)  # Choose the action with the highest Q-value

        self.last_action = current_action  # Update the last action

        return current_action

# Test Example
def test_greedy_agent():
    np.random.seed(1)
    greedy_agent = GreedyAgent()
    greedy_agent.q_values = [0, 0, 1.0, 0, 0]
    greedy_agent.arm_count = [0, 1, 0, 0, 0]
    greedy_agent.last_action = 1

    # Take a fake agent step
    action = greedy_agent.agent_step(reward=1)

    # Make sure agent took greedy action
    assert action == 2, "Agent did not take the greedy action."

    # Make sure q_values were updated correctly
    assert greedy_agent.q_values == [0, 0.5, 1.0, 0, 0], "Q-values not updated correctly after step 1."

    # Take another step
    action = greedy_agent.agent_step(reward=2)
    assert action == 2, "Agent did not take the greedy action."
    assert greedy_agent.q_values == [0, 0.5, 2.0, 0, 0], "Q-values not updated correctly after step 2."

# Run the test
test_greedy_agent()
