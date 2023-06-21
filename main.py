import math
from PIL import Image
import numpy as np
import time
import multiprocessing as mp

def normalize(vector):
    return vector / np.linalg.norm(vector)

def reflect(vector, normal):
    return vector - 2 * np.dot(vector, normal) * normal

def ray_sphere_intersection(ray_origin, ray_direction, sphere_center, sphere_radius):
    oc = ray_origin - sphere_center
    a = np.dot(ray_direction, ray_direction)
    b = 2.0 * np.dot(oc, ray_direction)
    c = np.dot(oc, oc) - sphere_radius**2
    discriminant = b**2 - 4*a*c

    if discriminant < 0:
        return None
    else:
        t1 = (-b - np.sqrt(discriminant)) / (2.0 * a)
        t2 = (-b + np.sqrt(discriminant)) / (2.0 * a)

        if t1 > 0:
            return t1
        elif t2 > 0:
            return t2
        else:
            return None
def random_hemisphere_direction(normal):
    u = np.random.rand()
    v = np.random.rand()
    
    theta = 2.0 * np.pi * u
    phi = np.arccos(2.0 * v - 1.0)
    
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    
    random_direction = np.array([x, y, z])
    
    # Transform the direction from the hemisphere coordinate system to the world coordinate system
    if random_direction.dot(normal) < 0:
        random_direction = -random_direction
    
    return random_direction
class Sphere:
    def __init__(self, center, radius, color, luminous=False):
        self.center = np.array(center)
        self.radius = radius
        self.luminous = luminous
        self.color = np.array([color])/255
class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction
        self.color = np.array([255,255,255])/255
        self.lit = False
        self.pos = []
        self.sphereHit = None
        self.bounces = 0
    def trace(self, spheres, max_bounces, reflectedRays):
        first_hit = None
        global tempoForSpheres
        global tempoForRays
        min_distance = float('inf')
        min_distance = float('inf')
        first_hit = None
        for sphere in spheres:
            hit_point, new_direction, new_color = self.reflect_ray(sphere)
            if hit_point is not None:
                distance = np.linalg.norm(hit_point - self.origin)
                if distance < min_distance:
                    min_distance = distance
                    first_hit = [hit_point, new_direction, new_color, sphere]
        if first_hit is not None:
            if first_hit[3].luminous and self.bounces == 0 and False: # True para fazer as luzes não refletirem
                return first_hit[3].color
            self.sphereHit, self.direction, self.origin, self.color = first_hit[3], first_hit[1], first_hit[0], first_hit[2]
            self.lit = self.lit or first_hit[3].luminous
            if max_bounces > 0:
                self.bounces += 1
                colors = np.zeros((reflectedRays, 3))
                for i in range(reflectedRays):
                    new_ray = Ray(self.origin, self.direction)
                    new_ray.color, new_ray.lit, new_ray.bounces = self.color, self.lit, self.bounces
                    color2 = new_ray.trace(spheres, max_bounces - 1, reflectedRays)
                    self.lit = new_ray.lit
                    colors[i] = color2 if new_ray.lit else np.array([0,0,0])
                self.color = colors.mean(axis=0)
        if self.lit:
            return self.color
        return np.array([0,0,0])

    def reflect_ray(self, sphere,):
        global tempoReflect
        t = ray_sphere_intersection(self.origin, self.direction, sphere.center, sphere.radius)
        if t is None:
            return None, None, None
        hit_point = self.origin + t * self.direction
        # Random direction in the hemisphere defined by the normal
        reflected_direction = random_hemisphere_direction(normalize(hit_point - sphere.center))
        return hit_point + 0.001 * reflected_direction, reflected_direction, self.color * sphere.color
def apply_rotation(direction, rotation):
    rx, ry, rz = rotation
    rot_x = np.array([[1, 0, 0],
                         [0, math.cos(rx), -math.sin(rx)],
                         [0, math.sin(rx), math.cos(rx)]])

    rot_y = np.array([[math.cos(ry), 0, math.sin(ry)],
                         [0, 1, 0],
                         [-math.sin(ry), 0, math.cos(ry)]])

    rot_z = np.array([[math.cos(rz), -math.sin(rz), 0],
                         [math.sin(rz), math.cos(rz), 0],
                         [0, 0, 1]])
    direction = direction.dot(rot_x).dot(rot_y).dot(rot_z)
    return direction

def render_row(args):
    y, width, height, spheres, max_bounces, reflectedRays, rotation, position, fov, aspect_ratio, scale = args
    row_pixels = []
    for x in range(width):
        ndc_x = (x + 0.5) / width
        ndc_y = (y + 0.5) / height

        screen_x = (2 * ndc_x - 1) * aspect_ratio * scale
        screen_y = (1 - 2 * ndc_y) * scale

        ray_direction = normalize(np.array([screen_x, screen_y, -1]))
        ray_direction = apply_rotation(ray_direction, rotation)
        offset = np.random.uniform(-0.0007, 0.0007, size=ray_direction.shape)
        ray = Ray(position, ray_direction)  # + offset)
        color = ray.trace(spheres, max_bounces, reflectedRays)
        row_pixels.append(tuple(map(int, (color.flatten() * 255).astype(int))))
    return row_pixels
class Camera:
    def __init__(self, position, rotation, width, height, fov):
        self.position = position
        self.rotation = rotation
        self.width = width
        self.height = height
        self.fov =math.radians(fov)

    def apply_rotation(self, direction):
        rx, ry, rz = self.rotation
        rot_x = np.array([[1, 0, 0],
                             [0, math.cos(rx), -math.sin(rx)],
                             [0, math.sin(rx), math.cos(rx)]])

        rot_y = np.array([[math.cos(ry), 0, math.sin(ry)],
                             [0, 1, 0],
                             [-math.sin(ry), 0, math.cos(ry)]])

        rot_z = np.array([[math.cos(rz), -math.sin(rz), 0],
                             [math.sin(rz), math.cos(rz), 0],
                             [0, 0, 1]])
        direction = direction.dot(rot_x).dot(rot_y).dot(rot_z)

        return direction
    def render(self, spheres, max_bounces, reflectedRays):
        aspect_ratio = self.width / self.height
        scale = math.tan(self.fov / 2)
        with mp.Pool() as pool:
            pixels = pool.map(render_row, [(y, self.width, self.height, spheres, max_bounces, reflectedRays, 
                                            self.rotation, self.position, self.fov, aspect_ratio, scale)
                                           for y in range(self.height)])
        image = Image.new("RGB", (self.width, self.height), "black")
        img_pixels = image.load()
        for y in range(self.height):
            for x in range(self.width):
                img_pixels[x, y] = pixels[y][x]
        return image
if __name__ == "__main__":
    bola = Sphere([100, 100, 0], 100, [255, 255, 255], True)
    bola2 = Sphere([0, 0, 290], 70, [230, 50, 250], True)
    bola3 = Sphere([0,0, 100], 200, [234, 135, 156])
    camera = Camera([400, 0, 200], [-1.5,0,-1.5], 1000, 500, 60)
    tempo = time.time()
    image = camera.render2([bola2, bola, bola3], 2, 1)
    print(time.time() - tempo)
    image.show()
