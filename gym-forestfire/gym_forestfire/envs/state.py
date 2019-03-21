import elements
import agent
from random import randint


class Environment:
	def __init__(self, width, height):
		self.running = True

		self.wind_speed = randint(0, 3)
		self.wind_vector = (randint(0, 1), randint(0, 1))

		self.width = width
		self.height = height
		self.world = self.create_world(width, height)
		
		self.burning_cells = set()
		self.barriers = set()
<<<<<<< HEAD:gym-forestfire/gym_forestfire/envs/state.py
		self.agents = list()

		self.total_fuel = self.get_total_fuel()
		self.fuel_burnt = 0
		
		self.agents.add(agent.Agent(5, 5, self))
		burning_cell = self.world[int(width/2)][int(height/2)]
		burning_cell.burning = True
		self.burning_cells.add(burning_cell)

	def reset_env(self):
		self.__init__(self.width, self.height)
=======
		self.agents = set()

		self.create_agent_and_fire()
>>>>>>> 1368f7d690abbdc95d4df622657614bda65df537:state.py

	def create_world(self, width, height):
		world = list()
		for x in range(width):
			line = list()
			for y in range(height):
				line.append(elements.Grass(x, y))
			world.append(line)
		return world

	def create_agent_and_fire(self):
		self.agents.add(agent.Agent(5, 5, self))
		burning_cell = self.world[int(self.width / 2)][int(self.height / 2)]
		burning_cell.burning = True
		self.burning_cells.add(burning_cell)

	def reset_world(self):
		self.running = True
		self.world = self.create_world(self.width, self.height)
		self.burning_cells = set()
		self.agents = set()
		self.create_agent_and_fire()

	def get_at(self, x, y):
		return self.world[x][y]

	def set_at(self, x, y, cell):
		self.world[x][y] = cell

	def get_agent_coords(self):
		coords = set()
		for agent in self.agents:
			coords.add((agent.x, agent.y))
		return coords

	def inbounds(self, x, y):
		return 0 <= x < self.width and 0 <= y < self.height

	def traversable(self, x, y):
		cell = self.get_at(x, y)
		return cell.type in ["Grass", "Dirt"] and not cell.burning

	# Take the still running check out of the update() method,
	# make an isTerminal() method which deals with this situation
	def update(self):
		if not self.agents or not self.burning_cells:
			self.running = False
			return

		temp_agents = set(self.agents)
		for a in self.agents:
			status = a.update()
			if status == "Dead":
<<<<<<< HEAD:gym-forestfire/gym_forestfire/envs/state.py
				agents_to_remove.add(agent)
		self.agents = self.agents - agents_to_remove

		self.fuel_burnt += len(self.burning_cells)
=======
				temp_agents.remove(a)
		self.agents = temp_agents
>>>>>>> 1368f7d690abbdc95d4df622657614bda65df537:state.py

		temp_burning_cells = set(self.burning_cells)
		for cell in self.burning_cells:
			status = cell.time_step()
			if status == "Burnt Out":
				temp_burning_cells.remove(cell)
			if status == "No Change":
				neighbours = cell.get_neighbours(self)
				for n_cell in neighbours:
					if n_cell.burnable:
						status = n_cell.get_heat_from(cell, self)
						if status == "Ignited":
							temp_burning_cells.add(n_cell)
		self.burning_cells = temp_burning_cells

	def get_total_fuel(self):
		total_fuel = 0
		for x in range(len(self.world)):
			for y in range(len(self.world[0])):
				total_fuel += self.get_at(x, y).fuel
		return total_fuel

	def get_fitness(self):
		return self.fuel_burnt

	def get_features(self):
		features = list()

		for a in self.agents:
			min_distance = distance = 1000
			closest_fire = None

			for cell in self.burning_cells:
				distance = cell.distance_to(a, "euclidean")
				if distance < min_distance:
					min_distance = distance
					closest_fire = cell

			(ax, ay) = a.get_pos()
			features.append(distance)
			features.append(closest_fire.angle_to(a))
			features.append(ax)
			features.append(ay)
			features.append(len(self.burning_cells))
