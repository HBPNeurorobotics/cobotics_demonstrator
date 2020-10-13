import imageio
import numpy as np
import random
import h5py
import time
import sys
import re
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from cdp4_data_collection import CDP4DataCollection
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from scipy.spatial.transform import Rotation
from cv2 import fillConvexPoly, fillPoly
from glob import glob

# Some global variables
class TrainingRoom():

	# May be this function should also build the walls and create the camera object
	def __init__(self, camera_params, image_dimensions, n_sequences_per_scene, n_frames_per_sequence=20):
		
		# Training_room parameters
		self.wall_height = 2.0
		self.wall_width = 0.1
		self.table_height = 0.78  # should check in sdf
		self.table_bounds_on  = {'x': [-0.60, 0.60], 'y': [-1.65, 0.35]}  # rough space for both tables
		self.table_bounds_off = {'x': [-0.60,-0.20], 'y': [-1.65,-0.35]}  # hole between both tables

		# Camera parameters
		self.image_dimensions = image_dimensions
		self.data_collector = CDP4DataCollection(camera_params)
		self.n_frames_per_sequence = n_frames_per_sequence
		self.n_cameras = camera_params['n_cameras']
		self.n_sequences_per_scene = n_sequences_per_scene
		self.n_sequences_per_camera = n_sequences_per_scene/self.n_cameras
		self.camera_type = camera_params['camera_type']
		self.camera_speed_states = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for c in range(self.n_cameras)]
		self.camera_speed_range_t = [0.03, 0.15]  # tangential speed
		self.camera_speed_range_r = [1.00, 1.00]  # radial speed, > 1 for inward spiral, < 1 for outward spiral
		self.camera_speed_t = np.random.uniform(self.camera_speed_range_t[0], self.camera_speed_range_t[1])
		self.camera_speed_r = np.random.uniform(self.camera_speed_range_r[0], self.camera_speed_range_r[1])
		self.camera_look_at = [0.2, -0.6, self.table_height]
		self.min_camera_radius = 0.5
		self.max_camera_radius = 1.0
		# ADD COLLISION TO TABLES AND REMOVE MY COLLISION CHECK

		# Objects in the scene variables
		self.object_presence_prob = [0.3, 0.7]
		self.object_shapes = ['table', 'eric', 'hammer', 'spanner', 'screwdriver', 'sphere', 'box', 'cylinder']
		self.object_numbers = {'table': 2, 'eric': 1, 'hammer': 5, 'spanner': 5, 'screwdriver': 5, 'sphere': 5, 'box': 5, 'cylinder': 5}
		self.object_min_scales = {'table': 1.0, 'eric': 1.0, 'hammer': 1.00, 'spanner': 1.50, 'screwdriver': 2.50, 'sphere': 0.30, 'box': 0.20, 'cylinder': 0.25}
		self.object_max_scales = {'table': 1.0, 'eric': 1.0, 'hammer': 0.25, 'spanner': 0.75, 'screwdriver': 1.25, 'sphere': 0.08, 'box': 0.05, 'cylinder': 0.06}
		self.object_names = ['_'.join((shape, str(n))) for shape in self.object_shapes for n in range(self.object_numbers[shape])]
		self.object_statics = [True if shape in ['table', 'eric'] else False for shape in self.object_shapes for n in range(self.object_numbers[shape])]
		self.object_instances = [i+1 for i, shape in enumerate(self.object_shapes) for n in range(self.object_numbers[shape])]  # 0 is no object
		self.object_poses = [self.data_collector.get_object_pose(n) for n in self.object_names]
		self.object_visibility = np.ones((len(self.object_names),), dtype=int)
		self.object_scales = [self.data_collector.get_object_scale(n) for n in self.object_names]
		self.vertex_index_meshes, self.triangle_index_meshes, self.posed_meshes, self.physical_scales = self.load_object_meshes()
		self.possible_colors = [[0,0,0], [0,255,0], [127,127,127], [0,0,255], [255,0,0]]

		# Classification and segmentation dataset variables
		data_images_shape = (self.n_sequences_per_scene, self.n_frames_per_sequence) + self.image_dimensions
		self.segment_images = 255*np.ones(data_images_shape, dtype=np.uint8)
		self.segment_labelling = np.zeros(self.image_dimensions[:-1], dtype=np.uint8)
		self.data_images = np.zeros(data_images_shape, dtype=np.uint8)
		self.data_labels = {'pos_3D': np.zeros((self.n_sequences_per_scene, len(self.object_names), 3)),
							'ori_3D': np.zeros((self.n_sequences_per_scene, len(self.object_names), 3)),
		                    'scale_3D': np.zeros((self.n_sequences_per_scene, len(self.object_names), 3)),
		                    'visibles': np.zeros((self.n_sequences_per_scene, len(self.object_names),)),
		                    'segments': np.zeros(data_images_shape[:-1], dtype=np.uint8)}

	# Select random object positions for the new scene
	def choose_new_object_poses(self):

		# Go through all moving objects to modify their positions and scales
		for (name, static, visible, pose, scale, phys_scale) in zip(self.object_names, self.object_statics,
			self.object_visibility, self.object_poses, self.object_scales, self.physical_scales):
			shape = name[:-2]
			if not static:  # some objects are always there

				# Set new positions for visble objects (checking for collision)
				if visible:
					bad_pos = True
					n_trials = 0
					while bad_pos and n_trials < 100:
						scale_x, scale_y, scale_z = np.random.uniform(
							self.object_min_scales[shape], self.object_max_scales[shape], (3,))
						if shape == 'cylinder':
							scale_y = scale_x
						if shape in ['sphere', 'spanner', 'screwdriver', 'hammer']:
							scale_y = scale_x
							scale_z = scale_x
						dest_x = np.random.uniform(self.table_bounds_on['x'][0], self.table_bounds_on['x'][1])
						dest_y = np.random.uniform(self.table_bounds_on['y'][0], self.table_bounds_on['y'][1])
						dest_z = phys_scale[-1]*scale_z/2 + 0.024 + self.table_height  # on the table
						bad_pos = False
						n_trials += 1
						bad_pos = self.check_bad_positions(name, dest_x, dest_y, scale_x, scale_y, phys_scale)
					if n_trials == 100:
						visible = False  # make object invisible if no good position is found
						self.object_visibility[self.object_names.index(name)] = False  # less weird way?
					else:
						dest_roll, dest_pitch, dest_yaw = 0.0, 0.0, 2*np.pi*np.random.random()

				# Set new positions for unvisible objects
				if not visible:  # no 'else' because some names might go through both conditions
					dest_x, dest_y, dest_z = 0.0, 0.0, -1.0
					scale_x, scale_y, scale_z = [self.object_min_scales[shape]]*3
					dest_roll, dest_pitch, dest_yaw = 0.0, 0.0, 0.0

				# Update new states
				pose.position.x = dest_x
				pose.position.y = dest_y
				pose.position.z = dest_z
				ori_4 = quaternion_from_euler(dest_roll, dest_pitch, dest_yaw)
				pose.orientation.x = ori_4[0]
				pose.orientation.y = ori_4[1]
				pose.orientation.z = ori_4[2]
				pose.orientation.w = ori_4[3]
				scale.x = scale_x
				scale.y = scale_y
				scale.z = scale_z

	# Check if new object would overlap with other existing objects
	def check_bad_positions(self, this_name, this_pos_x, this_pos_y, this_scale_x, this_scale_y, this_phys_scale):
		this_xmin = this_pos_x - this_scale_x*this_phys_scale[0]/2.0
		this_xmax = this_pos_x + this_scale_x*this_phys_scale[0]/2.0
		this_ymin = this_pos_y - this_scale_y*this_phys_scale[1]/2.0
		this_ymax = this_pos_y + this_scale_y*this_phys_scale[1]/2.0
		if (this_xmin < self.table_bounds_off['x'][1] and self.table_bounds_off['x'][0] < this_xmax\
			and this_ymin < self.table_bounds_off['y'][1] and self.table_bounds_off['y'][0] < this_ymax):
			return True
		# for (other_name, other_pose, other_scale, other_phys_scale, other_visible) in zip(
		# 	self.object_names, self.object_poses, self.object_scales, self.physical_scales, self.object_visibility):	
		# 	if other_name != this_name and other_visible:
		# 		other_xmin = other_pose.position.x - other_scale.x*other_phys_scale[0]/2.0
		# 		other_xmax = other_pose.position.x + other_scale.x*other_phys_scale[0]/2.0
		# 		other_ymin = other_pose.position.y - other_scale.y*other_phys_scale[1]/2.0
		# 		other_ymax = other_pose.position.y + other_scale.y*other_phys_scale[1]/2.0
		# 		if (this_xmin < other_xmax and other_xmin < this_xmax and this_ymin < other_ymax and other_ymin < this_ymax):
		# 			return True
		return False
			
	# Select new random position and angle for the camera
	def choose_new_camera_pose(self):
		radius = np.random.uniform(self.min_camera_radius, self.max_camera_radius)
		angle = 2*np.pi*np.random.random()
		dest_x = self.camera_look_at[0] + radius*np.cos(angle)
		dest_y = self.camera_look_at[1] + radius*np.sin(angle)
		dest_z = self.camera_look_at[2] + self.wall_height*np.random.uniform(0.2, 1.0)
		dest_roll = 0.0
		dest_pitch = np.arctan2(dest_z - self.camera_look_at[2], radius)  # points towards table plane
		dest_yaw = np.pi + angle
		return dest_x, dest_y, dest_z, dest_roll, dest_pitch, dest_yaw

	# Circular motion around the center
	def update_cameras_positions_and_speeds(self):
		for camera_id in range(self.n_cameras):

			# Update camera position and orientation according to its speeds
			v_x, v_y, v_z, v_roll, v_pitch, v_yaw = self.camera_speed_states[camera_id]
			self.data_collector.move_camera(camera_id, v_x, v_y, v_z, v_roll, v_pitch, v_yaw)

			# Update the speeds to obtain a circular (or spiralic) motion
			pose = self.data_collector.get_object_pose('camera_%02i' % (camera_id,))
			r_x = self.camera_look_at[0] - pose.position.x
			r_y = self.camera_look_at[1] - pose.position.y
			norm2_r = r_x**2 + r_y**2
			norm2_v = v_x**2 + v_y**2
			factor = norm2_v/norm2_r  # a_c = (v**2/r)*r_unit = (v**2/r**2)*r_vect
			v_x += self.camera_speed_r*factor*r_x
			v_y += self.camera_speed_r*factor*r_y
			if self.camera_speed_r != 1.0:  # keep same v_norm
				new_norm2_v = v_x**2 + v_y**2
				v_x = v_x*(norm2_v/new_norm2_v)**(0.5)
				v_y = v_y*(norm2_v/new_norm2_v)**(0.5)
			self.camera_speed_states[camera_id][0:2] = v_x, v_y

	# For each caemra, initiate new sequences of frames in a given scene
	def reset_cameras(self):
		for camera_id in range(self.n_cameras):

			# Set camera position and angles
			dest_x, dest_y, dest_z, dest_roll, dest_pitch, dest_yaw = self.choose_new_camera_pose()
			pose = self.data_collector.get_object_pose('camera_%02i' % (camera_id,))
			ori3_pose = euler_from_quaternion([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
			x = dest_x - pose.position.x
			y = dest_y - pose.position.y
			z = dest_z - pose.position.z
			roll = dest_roll - ori3_pose[0]
			pitch = dest_pitch - ori3_pose[1]
			yaw = dest_yaw - ori3_pose[2]
			self.data_collector.move_camera(camera_id, x, y, z, roll, pitch, yaw)

			# Compute the correct velocities to turn around the center
			self.camera_speed_t = np.random.uniform(self.camera_speed_range_t[0], self.camera_speed_range_t[1])
			self.camera_speed_r = np.random.uniform(self.camera_speed_range_r[0], self.camera_speed_range_r[1])
			sense = np.random.choice([-1, 1])
			r_x = dest_x - self.camera_look_at[0]
			r_y = dest_y - self.camera_look_at[1]
			v_x = self.camera_speed_t*sense*r_y
			v_y = -self.camera_speed_t*sense*r_x
			v_z = 0.0
			v_roll = 0.0
			v_pitch = 0.0
			norm_r = (r_x**2 + r_y**2)**(0.5)
			norm_v = (v_x**2 + v_y**2)**(0.5)
			v_yaw = -sense*(norm_v/norm_r)

			# Update speed variables of each camera
			# time.sleep(0.1/self.n_cameras)  # might be problematic, max be only need once per scene?
			self.camera_speed_states[camera_id] = [v_x, v_y, v_z, v_roll, v_pitch, v_yaw]

	# Re-shuffle all objects in the scene
	def reset_scene(self):
		self.object_visibility = np.random.choice([1, 0], p=self.object_presence_prob, size=(len(self.object_names),))
		self.object_visibility[self.object_statics] = 1  # some objects are always there
		self.choose_new_object_poses()
		for name, pose, scale in zip(self.object_names, self.object_poses, self.object_scales):
			self.data_collector.set_object_pose(name, pose, False)
			self.data_collector.set_object_scale(name, scale.x, scale.y, scale.z)

	# Update the labels after each scene reset (numpy format, special values for non-visible objects)
	def update_scene_labels(self):
		positions = [[p.position.x, p.position.y, p.position.z] for p in self.object_poses]
		orientations = [euler_from_quaternion([p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]) for p in self.object_poses]
		scales = [[s.x, s.y, s.z] for s in self.object_scales]
		for pos, ori, scl, visible in zip(positions, orientations, scales, self.object_visibility):
			if not visible:
				pos, ori, scl = ([-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0])
		self.data_labels['pos_3D'] = np.array([positions]*self.n_sequences_per_scene)
		self.data_labels['ori_3D'] = np.array([orientations]*self.n_sequences_per_scene)
		self.data_labels['scale_3D'] = np.array([scales]*self.n_sequences_per_scene)
		self.data_labels['visibles'] = np.array([self.object_visibility]*self.n_sequences_per_scene)

	# Create lists containing all voxel positions (and other useful arrays) inside every object
	def load_object_meshes(self, use_low_poly=True):
		all_vertices_meshes = {shape: np.zeros((0, 3), dtype=float) for shape in self.object_shapes}
		all_triangles_meshes = {shape: np.zeros((0, 3), dtype=int) for shape in self.object_shapes}
		all_physical_scales = {shape: None for shape in self.object_shapes}
		for shape in self.object_shapes:
			dae_file_path = './dae_files/%s_low.dae' % (shape,) if use_low_poly else './dae_files/%s.dae' % (shape,)
			with open(dae_file_path) as f:  # us
				s = f.read()
				vertices_info = re.findall(r'<float_array.+?mesh-positions-array.+?>(.+?)</float_array>', s)
				transform_info = re.findall(r'<matrix sid="transform">(.+?)</matrix>.+?<instance_geometry', s, flags=re.DOTALL)  # better way?
				triangles_info = re.findall(r'<triangles.+?<p>(.+?)</p>.+?</triangles>', s, flags=re.DOTALL)
				if len(triangles_info) == 0:
					triangles_info = re.findall(r'<polylist.+?<p>(.+?)</p>.+?</polylist>', s, flags=re.DOTALL)
				for part_id in range(len(vertices_info)):
					transform_matrix = np.array([float(n) for n in transform_info[part_id].split(' ')]).reshape((4, 4))
					vertices_temp = np.array([float(n) for n in vertices_info[part_id].split(' ')])
					vertices_temp = np.reshape(vertices_temp, (vertices_temp.shape[0]/3, 3))
					vertices_temp = np.dot(transform_matrix, np.c_[vertices_temp, np.ones(vertices_temp.shape[0])].T)[:-1].T
					triangles_temp = np.array([int(n) for n in triangles_info[part_id].split(' ')])[::3]
					triangles_temp = np.reshape(triangles_temp, (triangles_temp.shape[0]/3, 3))
					triangles_temp = triangles_temp + all_vertices_meshes[shape].shape[0]  # shift triangle indices
					all_vertices_meshes[shape] = np.vstack((all_vertices_meshes[shape], vertices_temp))
					all_triangles_meshes[shape] = np.vstack((all_triangles_meshes[shape], triangles_temp))
				min_pos = [all_vertices_meshes[shape][:, d].min() for d in range(3)]
				max_pos = [all_vertices_meshes[shape][:, d].max() for d in range(3)]
				correction = 2.0 if shape in ['sphere', 'box', 'cylinder'] else 1.0  # these .dae files actually do not match the actual objects
				all_physical_scales[shape] = [(max_pos[d] - min_pos[d])/correction for d in range(3)]

		# Better way?
		vertices_meshes_list = []
		triangles_meshes_list = []
		physical_scales_list = []
		posed_meshes_list = [None for name in self.object_names]
		for name in self.object_names:
			shape = name[:-2]
			vertices_meshes_list.append(all_vertices_meshes[shape])
			triangles_meshes_list.append(all_triangles_meshes[shape])
			physical_scales_list.append(all_physical_scales[shape])
		return (np.array(lst) for lst in [vertices_meshes_list, triangles_meshes_list, posed_meshes_list, physical_scales_list])

	# Transform the basic mesh coordinates with actual object psotion, scale and orientation
	def update_object_meshes(self):
		for i, (name, pos, scale, ori, visible) in enumerate(zip(self.object_names,
				self.data_labels['pos_3D'][0], self.data_labels['scale_3D'][0],
				self.data_labels['ori_3D'][0], self.data_labels['visibles'][0])):
			if visible:
				rot = Rotation.from_euler('z', ori[-1])  # or ('xyz', ori)
				scl = scale/2.0 if any([shape in name for shape in ['sphere', 'box', 'cylinder']]) else scale
				self.posed_meshes[i] = rot.apply(scl*self.vertex_index_meshes[i]) + pos

	# Compute distance of objects to the camera and return a sorted index list (furthest to closest)
	def sort_object_wrt_camera_distance(self, camera_id):
		cam = self.data_collector.cam_transform[camera_id]
		cam_pos = [cam.pos_x_m, cam.pos_y_m, cam.elevation_m]
		obj_positions = [[p.position.x, p.position.y, p.position.z] for p in self.object_poses]
		distances_to_cam = np.zeros((len(obj_positions),))
		for i, obj_pos in enumerate(obj_positions):
			distances_to_cam[i] = sum([(o-c)**2 for o, c in zip(obj_pos, cam_pos)])
		return distances_to_cam.argsort()[::-1]

	# Move camera around the scene and take screenshots
	def generate_data_subset(self):
		for sequence_id in range(self.n_sequences_per_camera):
			self.reset_cameras()
			for frame_id in range(self.n_frames_per_sequence):
				self.update_cameras_positions_and_speeds()
				for camera_id in range(self.n_cameras):
					sample_id = sequence_id*self.n_cameras + camera_id
					# if frame_id > 0:
					sequence_sample = self.data_collector.capture_image(camera_id, ensure_timing=True)
					self.data_images[sample_id, frame_id] = sequence_sample
					# if frame_id < self.n_frames_per_sequence - 1:
					self.segment_labelling[:] = 0
					dist_idxs = self.sort_object_wrt_camera_distance(camera_id)
					for (visible, vertices, triangles, dist_idx) in zip(self.object_visibility[dist_idxs],\
						self.posed_meshes[dist_idxs], self.triangle_index_meshes[dist_idxs], dist_idxs):
						if visible:
							color = self.possible_colors[dist_idx%len(self.possible_colors)]
							vertices_2D = np.array(self.data_collector.cam_transform[camera_id].imageFromSpace(vertices))
							if len(vertices_2D) > 0:
								segment_label_value = self.object_instances[dist_idx]
								triangles_2D = np.take(vertices_2D, triangles, axis=0).astype(int)

								# if segment_label_value == 2:
								# 	print(triangles_2D[triangles_2D[:,:,0] == [256, 256]])

								for triangle_2D in triangles_2D:
									fillConvexPoly(self.segment_labelling, triangle_2D, segment_label_value)
					self.data_labels['segments'][sample_id, frame_id] = self.segment_labelling
		if plot_segmentation:  # with last camera output (just to plot an example)
			self.record_segmentation_gif()

	# Save an example gif of object segmentation labelling (uses 1sr camera only)
	def record_segmentation_gif(self):
		segment_frames = []
		for frame_id in range(self.n_frames_per_sequence):
			fig, ax = plt.subplots(dpi=150)
			fig.subplots_adjust(hspace=0.5)
			plt.subplot(1,2,1)
			plt.title('Sample\nimage')
			plt.imshow(self.data_images[0, frame_id]/255.0)
			plt.axis('off')
			plt.subplot(1,2,2)
			plt.title('Segmentation\nlabelling')
			plt.imshow(self.data_labels['segments'][0, frame_id], vmin=0, vmax=max(self.object_instances))
			plt.axis('off')
			fig.canvas.draw()  # draw the canvas, cache the renderer
			segment_frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
			segment_frames.append(segment_frame.reshape(fig.canvas.get_width_height()[::-1] + (3,)))
			plt.close()
		imageio.mimsave('./segment_examples/sample_%02i.gif' % (scene_index+1,), segment_frames, fps=24)

# Generate the whole dataset
if __name__ == '__main__':

	# Camera and scene parameters
	camera_params = {'name': 'robot'}
	camera_params['camera_type'] = 'rgb'  # 'rgb' or 'dvs' THEN i WILL CHANGE TO BOTH
	camera_params['n_cameras'] = 1
	with open('../handover_virtual_room.sdf') as sdf:  # !!!all camera parameters should be the same!!!
		text = sdf.read()
		first_cam_text = text.split('<sensor name="camera_%s" type="camera">'
			%(camera_params['camera_type']))[1].split('</sensor>')[0]
		height = int(first_cam_text.split('<height>')[1].split('</height>')[0])
		width = int(first_cam_text.split('<width>')[1].split('</width>')[0])
		h_fov = float(first_cam_text.split('<horizontal_fov>')[1].split('</horizontal_fov>')[0])
		v_fov = h_fov*float(height)/float(width)
		n_cameras_max = int(text.split('<model name="camera_')[-1].split('">')[0])+1
	camera_params['camera_resolution'] = (width, height)
	camera_params['focal_length_px'] = 0.5003983220157445*width  # ??? not sure.. but it works
	if camera_params['n_cameras'] > n_cameras_max:
		print('Too many cameras selected: number of cameras set to %s.' % (n_cameras_max,))
		camera_params['n_cameras'] = n_cameras_max

	# Dataset parameters
	dataset_output_name = './training_room_dataset'  # a number is added to avoid overwriting
	plot_segmentation = True  # record segmentation labelling .gif examples
	n_color_channels = 3
	image_dimensions = camera_params['camera_resolution'] + (n_color_channels,)
	n_frames_per_sequence = 20
	n_sequences_per_scene = 1#6
	assert n_sequences_per_scene % camera_params['n_cameras'] == 0,\
		'Error: n_sequences_per_scene must be a multiple of n_cameras.'
	n_samples_per_dataset = 1000  # ~16 GB for uncompressed np.array((64000, 20, 64, 64, 3), dtype=np.uint8)
	n_scenes_per_dataset = int(n_samples_per_dataset/n_sequences_per_scene)
	if float(n_samples_per_dataset)/n_sequences_per_scene - n_scenes_per_dataset > 0:
		n_scenes_per_dataset += 1  # 1 partial run to finish the sequence samples
	training_room = TrainingRoom(camera_params, image_dimensions, n_sequences_per_scene, n_frames_per_sequence)

	# Create datasets to be filled by the NRP simulation
	starting_time = time.time()
	dataset_dims_image = (n_samples_per_dataset, n_frames_per_sequence,) + image_dimensions
	dataset_dims_labels_3D = (n_samples_per_dataset, len(training_room.object_names), 3)
	dataset_dims_labels_1D = (n_samples_per_dataset, len(training_room.object_names))
	dataset_dims_labels_seg = dataset_dims_image[:-1]
	chunk_dims_image = (1,) + dataset_dims_image[1:]
	chunk_dims_seg = (1,) + dataset_dims_image[1:-1]
	file_name_index = len(glob('%s_*.h5' % (dataset_output_name,))) + 1
	dataset_output_name = '%s_%02i.h5' % (dataset_output_name, file_name_index)
	with h5py.File(dataset_output_name, 'w') as f:
		f.create_dataset('image_samples', shape=dataset_dims_image, dtype='uint8', chunks=chunk_dims_image, compression='gzip')
		f.create_dataset('labels_pos_3D', shape=dataset_dims_labels_3D, dtype='float64')
		f.create_dataset('labels_ori_3D', shape=dataset_dims_labels_3D, dtype='float64')
		f.create_dataset('labels_scale_3D', shape=dataset_dims_labels_3D, dtype='float64')
		f.create_dataset('labels_visibles', shape=dataset_dims_labels_1D, dtype='int64')
		f.create_dataset('labels_segments', shape=dataset_dims_labels_seg, chunks=chunk_dims_seg, dtype='uint8', compression='gzip')

		# Fill the dataset with the generated sequences of frames and corresponding labels
		remaining_indexes = np.array(range(n_samples_per_dataset))
		for scene_index in range(n_scenes_per_dataset):
			first_id = scene_index*n_sequences_per_scene
			last_id = min((scene_index+1)*n_sequences_per_scene, n_samples_per_dataset)
			indexes_to_fill = np.random.choice(remaining_indexes, size=(last_id-first_id,), replace=False)
			remaining_indexes = np.delete(remaining_indexes, indexes_to_fill)
			sys.stdout.write('\rCreating dataset (%i/%i sequences generated)' % (first_id, n_samples_per_dataset))
			sys.stdout.flush()
			training_room.reset_scene()
			training_room.update_scene_labels()
			training_room.update_object_meshes()
			training_room.generate_data_subset()
			for i, sample_id in enumerate(indexes_to_fill):
				f['image_samples'][sample_id] = training_room.data_images[i]
				f['labels_pos_3D'][sample_id] = training_room.data_labels['pos_3D'][i]
				f['labels_ori_3D'][sample_id] = training_room.data_labels['ori_3D'][i]
				f['labels_scale_3D'][sample_id] = training_room.data_labels['scale_3D'][i]
				f['labels_visibles'][sample_id] = training_room.data_labels['visibles'][i]
				f['labels_segments'][sample_id] = training_room.data_labels['segments'][i]

	# Goodbye message
	n_minutes = int((time.time() - starting_time)/60) + 1
	print('\rDataset created in %i minutes (%i/%i sequences generated)' % (n_minutes, n_samples_per_dataset, n_samples_per_dataset))
