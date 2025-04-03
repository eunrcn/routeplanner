import random


possible_actions = { # k:v. original state: [lst of possible next states]
            "S1": ["S2", "S6", "S10"],
            "S2": ["S1", "S6", "S7", "S3", "S11"],
            "S3": ["S2", "S7", "S8", "S4"],
            "S4": ["S3", "S8", "S9", "S5"],
            "S5": ["S4", "S9", "S14"],
            "S6": ["S1", "S2", "S10", "S11", "S7"],
            "S7": ["S2", "S3", "S8", "S12", "S11", "S6"],
            "S8": ["S3", "S4", "S7", "S9", "S12", "S13"],
            "S9": ["S4", "S5", "S8", "S13", "S14"],
            "S10": ["S1", "S6", "S11"],
            "S11": ["S10", "S6", "S2", "S7", "S12"],
            "S12": ["S11", "S7", "S8", "S13"],
            "S13": ["S12", "S8", "S9", "S14"],
            "S14": ["S13", "S9", "S5"]
        }

################################################################################################
# Add weather score to states 
################################################################################################

# Weather exposure lookup dictionary
weather_exposure = {
    (1, 2): 95, (1, 6): 1, (1, 10): 1,
    (2, 1): 95, (2, 6): 5, (2, 7): 90, (2, 3): 44, (2, 11): 32,
    (3, 2): 44, (3, 7): 57, (3, 8): 9, (3, 4): 41,
    (4, 3): 41, (4, 8): 54, (4, 9): 50, (4, 5): 65,
    (5, 4): 65, (5, 9): 9, (5, 14): 52,
    (6, 1): 1, (6, 2): 5, (6, 10): 27, (6, 11): 50, (6, 7): 82,
    (7, 2): 90, (7, 3): 57, (7, 8): 51, (7, 12): 33, (7, 11): 27, (7, 6): 82,
    (8, 3): 9, (8, 4): 54, (8, 7): 51, (8, 9): 84, (8, 12): 36, (8, 13): 22,
    (9, 4): 50, (9, 5): 9, (9, 8): 84, (9, 13): 31, (9, 14): 39,
    (10, 1): 1, (10, 6): 27, (10, 11): 52,
    (11, 10): 52, (11, 6): 50, (11, 2): 32, (11, 7): 27, (11, 12): 18,
    (12, 11): 18, (12, 7): 33, (12, 8): 36, (12, 13): 46,
    (13, 12): 46, (13, 8): 22, (13, 9): 31, (13, 14): 80,
    (14, 13): 80, (14, 9): 39, (14, 5): 52
}

# Assign weather exposure scores for each action
weather_scores = {
    state: {action: weather_exposure.get((int(state[1:]), int(action[1:])), 1000)
            for action in neighbors}
    for state, neighbors in possible_actions.items()
}

# O(1) lookup for weather score
# define the state and action to retrieve the weather score
state, action = "S3", "S8"
weather_score = weather_scores[state][action]

################################################################################################
# Add safety score to states
################################################################################################

# Safety exposure lookup dictionary
safety_exposure = {
    (1, 2): 22, (1, 6): 13, (1, 10): 15,
    (2, 1): 22, (2, 6): 19, (2, 7): 36, (2, 3): 41, (2,11): 21,
    (3, 2): 41, (3, 7): 36, (3, 8): 42, (3, 4): 48,
    (4, 3): 48, (4, 8): 42, (4, 9): 62, (4, 5): 66,
    (5, 4): 66, (5, 9): 99, (5, 14): 99,
    (6, 1): 13, (6, 2): 19, (6, 10): 0, (6, 11): 18, (6, 7): 21,
    (7, 2): 36, (7, 3): 36, (7, 8): 38, (7, 12): 39, (7, 11): 18, (7, 6): 21,
    (8, 3): 42, (8, 4): 42, (8, 7): 38, (8, 9): 64, (8, 12): 58, (8, 13): 69,
    (9, 4): 62, (9, 5): 99, (9, 8): 64, (9, 13): 58, (9, 14): 99,
    (10, 1): 15, (10, 6): 0, (10, 11): 1,
    (11, 10): 1, (11, 6): 18, (11, 2): 21, (11, 7): 18, (11, 12): 46,
    (12, 11): 46, (12, 7): 39, (12, 8): 58, (12, 13): 69,
    (13, 12): 69, (13, 8): 69, (13, 9): 58, (13, 14): 99,
    (14, 13): 99, (14, 9): 99, (14, 5): 99
}

# Assign safety scores for each action
safety_scores = {
    state: {action: safety_exposure.get((int(state[1:]), int(action[1:])), 1000)
            for action in neighbors}
    for state, neighbors in possible_actions.items()
}

# O(1) lookup for safety score
# define the state and action to retrieve the safety score
state, action = "S3", "S8"
safety_score = safety_scores[state][action]

################################################################################################
# Add travel time score to states
################################################################################################

# Travel Time lookup dictionary
travel_time = {
    (1, 2): 20, (1, 6): 80, (1, 10): 100,
    (2, 1): 20, (2, 6): 60, (2, 7): 60, (2, 3): 40, (2, 11): 100,
    (3, 2): 40, (3, 7): 60, (3, 8): 60, (3, 4): 20,
    (4, 3): 20, (4, 8): 60, (4, 9): 60, (4, 5): 20,
    (5, 4): 20, (5, 9): 60, (5, 14): 100,
    (6, 1): 80, (6, 2): 60, (6, 10): 60, (6, 11): 40, (6, 7): 20,
    (7, 2): 60, (7, 3): 60, (7, 8): 20, (7, 12): 60, (7, 11): 60, (7, 6): 20,
    (8, 3): 60, (8, 4): 60, (8, 7): 20, (8, 9): 20, (8, 12): 60, (8, 13): 40,
    (9, 4): 60, (9, 5): 60, (9, 8): 20, (9, 13): 60, (9, 14): 60,
    (10, 1): 100, (10, 6): 60, (10, 11): 20,
    (11, 10): 20, (11, 6): 40, (11, 2): 100, (11, 7): 60, (11, 12): 20,
    (12, 11): 20, (12, 7): 60, (12, 8): 60, (12, 13): 20,
    (13, 12): 20, (13, 8): 40, (13, 9): 60, (13, 14): 20,
    (14, 13): 20, (14, 9): 60, (14, 5): 100
}

# Assign travel time scores for each action
travel_time_scores = {
    state: {action: travel_time.get((int(state[1:]), int(action[1:])), 1000)
            for action in neighbors}
    for state, neighbors in possible_actions.items()
}

# O(1) lookup for travel time score
# define the state and action to retrieve the travel time score
state, action = "S3", "S8"
travel_time_score = travel_time_scores[state][action]

##################################### Pedestrian Path class

class PedestrianPaths:
    def __init__(self, weather, safety, travel):
        self.state = "S1"  # Start state
        self.goal = "S14"
        self.num_states = 14
        self.possible_actions = { # k:v. original state: [lst of possible next states]
            "S1": ["S2", "S6", "S10"],
            "S2": ["S1", "S6", "S7", "S3", "S11"],
            "S3": ["S2", "S7", "S8", "S4"],
            "S4": ["S3", "S8", "S9", "S5"],
            "S5": ["S4", "S9", "S14"],
            "S6": ["S1", "S2", "S10", "S11", "S7"],
            "S7": ["S2", "S3", "S8", "S12", "S11", "S6"],
            "S8": ["S3", "S4", "S7", "S9", "S12", "S13"],
            "S9": ["S4", "S5", "S8", "S13", "S14"],
            "S10": ["S1", "S6", "S11"],
            "S11": ["S10", "S6", "S2", "S7", "S12"],
            "S12": ["S11", "S7", "S8", "S13"],
            "S13": ["S12", "S8", "S9", "S14"],
            "S14": ["S13", "S9", "S5"]
        }
        self.transition_probabilities = {
            "S1": 0.6,
            "S2": 0.6,
            "S3": 0.6,
            "S4": 0.6,
            "S5": 0.6,
            "S6": 0.7,
            "S7": 0.7,
            "S8": 0.7,
            "S9": 0.7,
            "S10": 0.7,
            "S11": 0.8,
            "S12": 0.8,
            "S13": 0.9,
            "S14": 0.9
        }
        self.weather = weather
        self.safety = safety
        self.travel = travel
        # Define the normalization ranges (assuming max and min possible values for each score type)
        self.WEATHER_MAX = 1000
        self.SAFETY_MAX = 1000
        self.TRAVEL_TIME_MAX = 1000
        # Define the weights for each score (adjust based on their importance)
        self.WEATHER_WEIGHT = 0.3
        self.SAFETY_WEIGHT = 0.4
        self.TRAVEL_TIME_WEIGHT = 0.3

    def step(self, action): #str -> (str, int, bool) #simple since deterministic action
        """ Move in the environment based on action """
        probability = self.transition_probabilities[action]
        if random.random() < probability: #successfully found the way to this state
            self.state = action
        #else, got lost and remain in the same state
        
        # Define rewards
        reward = 1 if self.state == self.goal else -0.3
        done = self.state == self.goal  # Episode ends when goal is reached
        return self.state, reward, done

    def reset(self): # _ -> str
        self.state = "S1"
        return self.state
    
################################################################################################
# Functions to dynamically calculate the reward based on the state and action, using the scores (also based on the (weather, safety, travel time) weights depending on what user prioritizes most?)
################################################################################################
    def normalize_score(self, score, max_score): #int,int -> int
        """ Normalize the score based on the maximum score for each category. """
        return score / max_score


    def calculate_reward(self, state, action, weather_weight, safety_weight, travel_time_weight): # -> int
        """ 
        Smarter reward function that considers normalized scores and weighted importance. 
        Incorporates dynamic scaling for weather, safety, and travel time.
        """

        # Retrieve individual scores
        weather_score = self.weather_scores[state][action]
        safety_score = self.safety_scores[state][action]
        travel_time_score = self.travel_time_scores[state][action]

        # Normalize the scores to be between 0 and 1
        normalized_weather = self.normalize_score(weather_score, self.WEATHER_MAX)
        normalized_safety = self.normalize_score(safety_score, self.SAFETY_MAX)
        normalized_travel_time = self.normalize_score(
            travel_time_score, self.TRAVEL_TIME_MAX)

        # Calculate weighted sum of normalized scores
        total_score = (self.WEATHER_WEIGHT * normalized_weather +
                    self.SAFETY_WEIGHT * normalized_safety +
                    self.TRAVEL_TIME_WEIGHT * normalized_travel_time)

        # Reward is the negative of the total score (since higher scores should be worse)
        reward = -1 * total_score

        return reward

env = PedestrianPaths(weather_scores, safety_scores, travel_time_scores)