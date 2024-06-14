import math
import csv
import random
import time
import numpy as np
from sklearn.linear_model import SGDClassifier

class Object:
    def __init__(self, x, y, direction):
        self.x = round(x, 2)
        self.y = round(y, 2)
        self.direction = direction
        
        
class World:
    def __init__(self, width, height, obstacle_color, target_color):
        self.width = width
        self.height = height
        self.obstacle_color = obstacle_color
        self.target_color = target_color
        self.target_x = round(self.width * np.random.rand(), 2)
        self.target_y = round(self.height * np.random.rand(), 2)
        self.obstacles = []

    def add_obstacle(self, x, y):
        self.obstacles.append((round(x, 2), round(y, 2)))

    def get_color_at_position(self, position):
        if position[0] < 0 or position[0] > self.width or position[1] < 0 or position[1] > self.height:
            return self.obstacle_color
        if position in self.obstacles:
            return self.obstacle_color
        return None

class Agent(Object):
    def __init__(self, x, y, direction, world,info):
        super().__init__(x, y, direction)
        self.view_distance = 2
        self.perception_range = 15
        self.classifier = SGDClassifier()
        self.prediction_module = PredictionModule()
        self.optimization_module = OptimizationModule()
        self.goal_x = round(world.width * np.random.rand(), 2)
        self.goal_y = round(world.height * np.random.rand(), 2)
        self.exactTargetX = world.target_x
        self.exactTargetY = world.target_y
        self.info = info
    

    def sense(self, world):
        def detect_color(world, position):
            return world.get_color_at_position(position)

        def calculate_relative_position(agent_position, object_position):
            return (round(object_position[0] - agent_position[0], 2), round(object_position[1] - agent_position[1], 2))

        def sense_world(agent_position, vision_range, world):
            detected_objects = {
                'obstacles': [],
                'target': None
            }

            for angle in range(360):
                for distance in range(1, vision_range + 1):
                    object_position = (
                        round(agent_position[0] + distance * math.cos(math.radians(angle)), 2),
                        round(agent_position[1] + distance * math.sin(math.radians(angle)), 2)
                    )

                    color = detect_color(world, object_position)

                    if color == world.obstacle_color:
                        relative_position = calculate_relative_position(agent_position, object_position)
                        if 0 <= relative_position[0] <= 200 and 0 <= relative_position[1] <= 100:
                            detected_objects['obstacles'].append(relative_position)
                            self.prediction_module.update_experience(relative_position, None)
                            break
                    elif color == world.target_color:
                        relative_position = calculate_relative_position(agent_position, object_position)
                        if 0 <= relative_position[0] <= 200 and 0 <= relative_position[1] <= 100:
                            detected_objects['target'] = relative_position
                            self.prediction_module.update_experience(None, relative_position)
                            break

            return detected_objects

        sensed_data = sense_world((self.x, self.y), self.view_distance, world)
        return sensed_data

    def make_decision(self, sensed_data):
        obstacle_prediction = 0
        target_prediction = 0

        if sensed_data['target'] is not None:
            prediction_result = self.prediction_module.prediction(sensed_data)
            target_prediction = prediction_result['target_prediction']

        if sensed_data['obstacles']:
            prediction_result = self.prediction_module.prediction(sensed_data)
            obstacle_prediction = prediction_result['obstacle_prediction']

        if obstacle_prediction == 1:
            return 'avoid_obstacle'
        elif target_prediction == 1:
            return 'move_towards_target'
        else:
            return 'move_towards_goal'

    def learn(self, sensed_data):
        if sensed_data['target'] is not None:
            label = 1
        else:
            label = 0

        if sensed_data['obstacles']:
            X_train_obstacle = np.array(sensed_data['obstacles'])
            y_train_obstacle = np.full(X_train_obstacle.shape[0], label)

            y_train_obstacle = y_train_obstacle.flatten()
            self.prediction_module.update_experience(X_train_obstacle, None)

        if sensed_data['target'] is not None:
            X_train_target = np.array(sensed_data['target'])
            y_train_target = np.full(X_train_target.shape[0], label)

            y_train_target = y_train_target.flatten()
            self.prediction_module.update_experience(None, X_train_target)

    def move(self, decision, sensed_data):
        self.x = round(self.x,2)
        self.y = round(self.y,2)
        if self.info == 1:
            self.move_towards(self.exactTargetX, self.exactTargetY, sensed_data)
        else:
            if decision == 'avoid_obstacle':
                self.avoid_obstacle()
            elif decision == 'move_towards_target':
               
                detected_objects = self.sense(self.world)
                path, success = self.optimization_module.decision(detected_objects, self.optimization_module,
                                                                (self.goal_x, self.goal_y))

                if not success:
                    path = self.optimization_module.optimize_path(path)
                    self.optimization_module.adjust_parameters(self.optimization_module)
                    success_rate = self.optimization_module.feedback_loop(path)

                    if success_rate < self.optimization_module.predefined_threshold:
                        self.optimization_module.trial_and_error_method(self.optimization_module, detected_objects,
                                                                        (self.goal_x, self.goal_y))
                    else:
                        self.optimization_module.update(path, success)
                self.x = round(self.x,2)
                self.y = round(self.y,2)
                self.x, self.y = round(path[-1][0], 2), round(path[-1][1], 2)
            else:
                if sensed_data['target'] is not None:
                    self.move_towards(self.goal_x, self.goal_y, sensed_data)
                else:
                    self.random_movement()


    def distance_to(self, x, y):
        return round(np.sqrt((self.x - x) ** 2 + (self.y - y) ** 2), 2)

    def move_towards(self, target_x, target_y, sensed_data):
        
        self.x = round(self.x,2)
        self.y = round(self.y,2)
        predicted_target_position = None
        if sensed_data['target'] is not None:
            predicted_target_position = self.prediction_module.predict_target((self.x, self.y))

        if predicted_target_position is not None:
            target_x, target_y = predicted_target_position

        if abs(self.x - self.goal_x) <= 6.5 and abs(self.y - self.goal_y) <= 7.5 and (abs(self.x - self.exactTargetX) > 6.5 or abs(self.y - self.exactTargetY) > 7.5 ):
            self.goal_x  = round(100 * np.random.rand(), 2)
            self.goal_y = round(200 * np.random.rand(), 2)

        angle_to_target = np.arctan2(target_y - self.y, target_x - self.x)
        distance_to_target = self.distance_to(target_x, target_y)

        if distance_to_target > 0.1:
            self.x += round(3 * np.cos(angle_to_target), 2)
            self.y += round(3 * np.sin(angle_to_target), 2)
        else:
            self.x = round(target_x, 2)
            self.y = round(target_y, 2)

    def avoid_obstacle(self):
        self.x = round(self.x,2)
        self.y = round(self.y,2)
        self.x -= round(np.cos(self.direction), 2)
        self.y -= round(np.sin(self.direction), 2)

    def random_movement(self):
        self.direction = round(np.random.rand() * 2 * np.pi, 2)
        self.x = round(self.x,2)
        self.y = round(self.y,2)
        self.x += round(np.cos(self.direction), 2)
        self.y += round(np.sin(self.direction), 2)


class PredictionModule:
    def __init__(self):
        self.obstacle_classifier = SGDClassifier(random_state=42)
        self.target_classifier = SGDClassifier(random_state=42)
        self.obstacle_training_data = np.empty((0, 200))  

        self.target_training_data = np.empty((0, 200))  

        self.obstacle_labels = []
        self.target_labels = []

    def train_obstacle_classifier(self):
        X = self.obstacle_training_data  
        y = np.array(self.obstacle_labels)
        self.obstacle_classifier.partial_fit(X, y, classes=[0, 1])

    def train_target_classifier(self):
        X = self.target_training_data

        y = np.array(self.target_labels)
        self.target_classifier.partial_fit(X, y, classes=[0, 1])


    def update_experience(self, obstacle_coordinates, target_coordinates):
        if obstacle_coordinates is not None:
            for coord in obstacle_coordinates:
                features = np.zeros(200)
                features[:len(coord)] = coord 
                self.obstacle_training_data = np.vstack([self.obstacle_training_data, features])
                self.obstacle_labels.append(1)

        if target_coordinates is not None:
            for coord in target_coordinates:
                features = np.zeros(200)
                features[:len(coord)] = coord 
                self.target_training_data = np.vstack([self.target_training_data, features])
                self.target_labels.append(1)
                
    def predict_obstacles(self, current_features):
        self.train_obstacle_classifier()
        predicted_obstacle = self.obstacle_classifier.predict([current_features])
        return predicted_obstacle[0]

    def predict_target(self, current_features):
        self.train_target_classifier()
        predicted_target = self.target_classifier.predict([current_features])
        return predicted_target[0]

    def prediction(self, input_data):
        self.train_obstacle_classifier()
        self.train_target_classifier()

        obstacle_prediction = self.predict_obstacles(input_data['obstacles'])
        target_prediction = self.predict_target(input_data['target'])

        prediction_result = {
            'obstacle_prediction': obstacle_prediction,
            'target_prediction': target_prediction
        }
        return prediction_result


class OptimizationModule:
    def __init__(self):
        self.predefined_threshold = 0.7

    def decision(self, detected_objects, learning_module, target):
        path = learning_module.decide_best_path(detected_objects, target)
        success = learning_module.evaluate_path(path, detected_objects)
        return path, success

    def optimize_path(self, path):
        np.random.shuffle(path)
        return path

    def adjust_parameters(self, learning_module):
        learning_module.predefined_threshold += round(np.random.randn() * 0.01, 2)

    def feedback_loop(self, path):
        return round(np.random.random(), 2)

    def trial_and_error_method(self, learning_module, detected_objects, target):
        new_path = learning_module.create_new_path(detected_objects, target)
        success = learning_module.evaluate_path(new_path, detected_objects)
        learning_module.update(new_path, success)

    def optimization(self, agent, learning_module, target):
        agent_active = True

        while agent_active:
            detected_objects = agent.sense_surroundings()
            path, success = self.decision(detected_objects, learning_module, target)

            if not success:
                path = self.optimize_path(path)
                self.adjust_parameters(learning_module)
                success_rate = self.feedback_loop(path)

                if success_rate < self.predefined_threshold:
                    self.trial_and_error_method(learning_module, detected_objects, target)
                else:
                    learning_module.update(path, success)




    
world = World(100, 200, 'red', 'green')
agent = Agent(round(np.random.rand() * 100, 2), round(np.random.rand() * 200, 2), np.random.rand() * 2 * np.pi, world,0)
agent.goal_x = round(agent.goal_x, 2)
agent.goal_y = round(agent.goal_y, 2)
        

agent.info = 0
agent.goal_x = round(np.random.rand() * 100, 2)
agent.goal_y = round(np.random.rand() * 200, 2)

    
for _ in range(60):
         
    x = round(np.random.rand() * 100, 2)
    y = round(np.random.rand() * 200, 2)
    new_obstacle = (x, y)

       
    while new_obstacle in world.obstacles:
            
        x = round(np.random.rand() * 100, 2)
        y = round(np.random.rand() * 200, 2)
        new_obstacle = (x, y)

            
    world.add_obstacle(x, y)

for i in range(100):
    sensed_data = agent.sense(world)
    decision = agent.make_decision(sensed_data)
    agent.move(decision,sensed_data)
    agent.learn(sensed_data)
    agent.x = round(agent.x,2)
    agent.y = round(agent.y,2)


    if i == 0:
        print(f"Hedef Konumu: x={world.target_x}, y={world.target_y}")

    if sensed_data['target'] is not None and i == 0:
        goal_x, goal_y = sensed_data['target']
        agent.set_goal(world.target_x, world.target_y)
        print(f"Belirlenen Hedef Konumu: x={goal_x}, y={goal_y}")


    print(f"Agent Konumu: x={agent.x}, y={agent.y}")

            
    if abs(agent.x - world.target_x) <= 6.5 and abs(agent.y - world.target_y) <= 7.5:
        print("Başarı! Agent belirlenen hedefe ulaştı.")
        print(i)
        break






