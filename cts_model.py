import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc as misc
import sys
import time
import torch

import cts.model as model

def rgb_to_symbol(channels):
    """Converts an RGB triple into an atomic colour (24-bit integer).
    """
    return (channels[0] << 16) | (channels[1] << 8) | (channels[2] << 0)

def symbol_to_rgb(colour):
    """Inverse operation of rgb_to_symbol.
    
    Returns: a (r, g, b) tuple.
    """
    return (colour >> 16, (colour >> 8) & 0xFF, (colour >> 0) & 0xFF)

def rgb_to_symbols(frame, output):
    """Preprocesses the given frame into a CTS-compatible representation.
    """
    assert(frame.shape[:2] == output.shape)
    for y in range(frame.shape[0]):
        for x in range(frame.shape[1]):
            output[y, x] = rgb_to_symbol(frame[y, x])

    return output

def symbols_to_rgb(frame, output):
    """Inverse of rgb_to_symbols.
    """
    assert(frame.shape == output.shape[:2])
    for y in range(frame.shape[0]):
        for x in range(frame.shape[1]):
            output[y, x] = symbol_to_rgb(frame[y, x])

    return output

class ConvolutionalMarginalDensityModel(object):
    """A density model for Freeway frames."""
    def __init__(self, frame_shape):
        """Constructor.
        
        Args:
            init_frame: A sample frame (numpy array) from which we determine the shape and type of our data.
        """
        self.convolutional_model = model.CTS(context_length=0)
        self.frame_shape = frame_shape
        
    def update(self, frame):
        assert(frame.shape == self.frame_shape)
        
        total_log_probability = 0.0
        # We simply apply the CTS update to each pixel.
        for y in range(frame.shape[0]):
            for x in range(frame.shape[1]):
                # Convert all 3 channels to an atomic colour.
                colour = rgb_to_symbol(frame[y, x])
                total_log_probability += self.convolutional_model.update(context=[], symbol=colour)

        return total_log_probability

    def sample(self):
        output_frame = np.zeros(self.frame_shape, dtype=np.uint8)

        for y in range(output_frame.shape[0]):
            for x in range(output_frame.shape[1]):
                # Use rejection sampling to avoid generating non-Atari colours.
                colour = self.convolutional_model.sample(context=[], rejection_sampling=True)
                output_frame[y, x] = symbol_to_rgb(colour)
        
        return output_frame

#version with symbol frame
# class ConvolutionalDensityModel(object):
#     """A density model for Freeway frames.
    
#     This one predict according to an L-shaped context around the current pixel.
#     """
#     def __init__(self, frame_shape, context_functor, alphabet=None):
#         """Constructor.
        
#         Args:
#             init_frame: A sample frame (numpy array) from which we determine the shape and type of our data.
#             context_functor: Function mapping image x position to a context.
#         """
#         # For efficiency, we'll pre-process the frame into our internal representation.
#         self.symbol_frame = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.int32) #it was uint32
#         self.symbol_frame=torch.tensor(self.symbol_frame, device='cuda', dtype=torch.uint8) #try int32
#         context_length = len(context_functor(self.symbol_frame, -1, -1))
#         self.convolutional_model = model.CTS(context_length=context_length, alphabet=alphabet)
        
#         self.context_functor = context_functor
#     def update(self, frame):
#         #print('self.symbol_frame',self.symbol_frame.shape)
#         #print('frame',frame.shape)
#         rgb_to_symbols(frame, self.symbol_frame)
#         print('self.symbol_frame',self.symbol_frame)
        
#         total_log_probability = 0.0
#         for y in range(self.symbol_frame.shape[0]):
#             for x in range(self.symbol_frame.shape[1]):
#                 context = self.context_functor(self.symbol_frame, y, x)
#                 colour = self.symbol_frame[y, x]
#                 total_log_probability += self.convolutional_model.update(context=context, symbol=colour)

#         return total_log_probability
#     def query(self, frame):
#         rgb_to_symbols(frame, self.symbol_frame)
#         total_log_probability = 0.0
#         for y in range(self.symbol_frame.shape[0]):
#             for x in range(self.symbol_frame.shape[1]):
#                 context = self.context_functor(self.symbol_frame, y, x)
#                 colour = self.symbol_frame[y, x]
#                 total_log_probability += self.convolutional_model.log_prob(context=context, symbol=colour)
#         return total_log_probability

#     def sample(self):
#         output_frame = np.zeros((*self.symbol_frame.shape, 3), dtype=np.uint8)
#         #print (output_frame.shape)
        
#         for y in range(self.symbol_frame.shape[0]):
#             for x in range(self.symbol_frame.shape[1]):
#                 # From a programmer's perspective, this is why we must respect the chain rule: otherwise
#                 # we condition on garbage.
#                 context = self.context_functor(self.symbol_frame, y, x)
#                 self.symbol_frame[y, x] = self.convolutional_model.sample(context=context,
#                                                                           rejection_sampling=True)

#         return symbols_to_rgb(self.symbol_frame, output_frame)

#     def sample_with_update(self):
#         output_frame = np.zeros((*self.symbol_frame.shape, 3), dtype=np.uint8)
#         #print (output_frame.shape)
        
#         for y in range(self.symbol_frame.shape[0]):
#             for x in range(self.symbol_frame.shape[1]):
#                 context = self.context_functor(self.symbol_frame, y, x)
#                 self.symbol_frame[y, x] = self.convolutional_model.sample(context=context,
#                                                                           rejection_sampling=False)
#                 self.convolutional_model.update(context, self.symbol_frame[y, x])
    
#         return symbols_to_rgb(self.symbol_frame, output_frame)
def dilations_context(image, y, x):
    """Generates a dilations-based context.
    
    We successively dilate first to the left, then up, then diagonally, with strides 1, 2, 4, 8, 16.
    """
    SPAN = 5
    # Default to -1 context.
    context = [0xFFFFFFFF] * (SPAN * 3)
    
    min_x, index = 1, (SPAN * 3) - 1
    for i in range(SPAN):
        if x >= min_x:
            context[index] = image[y][x - min_x]
        index -= 3
        min_x = min_x << 1

    min_y, index = 1, (SPAN * 3) - 2
    for i in range(SPAN):
        if y >= min_y:
            context[index] = image[y - min_y][x]
        index -= 3
        min_y = min_y << 1

    min_p, index = 1, (SPAN * 3) - 3
    for i in range(SPAN):
        if x >= min_p and y >= min_p:
            context[index] = image[y - min_p][x - min_p]
        index -= 3
        min_p = min_p << 1

    return context

def L_shaped_context(image, y, x):
    """This grabs the L-shaped context around a given pixel.
    
    Out-of-bounds values are set to 0xFFFFFFFF."""
    context = [0xFFFFFFFF] * 4
    if x > 0:
        context[3] = image[y][x - 1]
    
    if y > 0:
        context[2] = image[y - 1][x]
        context[1] = image[y - 1][x - 1] if x > 0 else 0
        context[0] = image[y - 1][x + 1] if x < image.shape[1] - 1 else 0

    # The most important context symbol, 'left', comes last.
    return context

#the version of Marc Bellemare
    
class LocationDependentDensityModel(object):
    """A density model for Freeway frames.
    
    This is exactly the same as the ConvolutionalDensityModel, except that we use one model for each
    pixel location.
    """
    def __init__(self, frame_shape, context_functor, alphabet=None):
        """Constructor.
        
        Args:
            init_frame: A sample frame (numpy array) from which we determine the shape and type of our data.
            context_functor: Function mapping image x position to a context.
        """
        # For efficiency, we'll pre-process the frame into our internal representation.
        self.symbol_frame = np.zeros((frame_shape[0:2]), dtype=np.uint32)

        context_length = len(context_functor(self.symbol_frame, -1, -1))
        self.models = np.zeros(frame_shape[0:2], dtype=object)
        
        for y in range(frame_shape[0]):
            for x in range(frame_shape[1]):
                self.models[y, x] = model.CTS(context_length=context_length, alphabet=alphabet)
        
        self.context_functor = context_functor

    def update(self, frame):
        #print('frame shape',frame.shape)
        #print('symbol frame',self.symbol_frame.shape)
        rgb_to_symbols(frame, self.symbol_frame)
        
        total_log_probability = 0.0
        for y in range(self.symbol_frame.shape[0]):
            for x in range(self.symbol_frame.shape[1]):
                context = self.context_functor(self.symbol_frame, y, x)
                colour = self.symbol_frame[y, x]
                total_log_probability += self.models[y, x].update(context=context, symbol=colour)

        return total_log_probability
    
    def query(self, frame):
        rgb_to_symbols(frame, self.symbol_frame)
        total_log_probability = 0.0
        for y in range(self.symbol_frame.shape[0]):
            for x in range(self.symbol_frame.shape[1]):
                context = self.context_functor(self.symbol_frame, y, x)
                colour = self.symbol_frame[y, x]
                total_log_probability += self.models[y, x].log_prob(context=context, symbol=colour)
        return total_log_probability

    def sample(self):
        output_frame = np.zeros((*self.symbol_frame.shape, 3), dtype=np.uint8)
        #print (output_frame.shape)
        
        for y in range(self.symbol_frame.shape[0]):
            for x in range(self.symbol_frame.shape[1]):
                # From a programmer's perspective, this is why we must respect the chain rule: otherwise
                # we condition on garbage.
                context = self.context_functor(self.symbol_frame, y, x)
                self.symbol_frame[y, x] = self.models[y, x].sample(context=context, rejection_sampling=True)

        return symbols_to_rgb(self.symbol_frame, output_frame)

#version without frame_to_symbol
# class LocationDependentDensityModel(object):
#     """A density model for Freeway frames.

#     This is exactly the same as the ConvolutionalDensityModel, except that we use one model for each
#     pixel location.
#     """

#     def __init__(self, frame_shape, context_functor, alphabet=None):
#         """Constructor.

#         Args:
#             init_frame: A sample frame (numpy array) from which we determine the shape and type of our data.
#             context_functor: Function mapping image x position to a context.
#         """
#         # For efficiency, we'll pre-process the frame into our internal representation.
#         self.frame_shape = frame_shape
#         context_length = len(context_functor(np.zeros((frame_shape[0:2]), dtype=np.uint32), -1, -1))
#         self.models = np.zeros(frame_shape[0:2], dtype=object)

#         for y in range(frame_shape[0]):
#             for x in range(frame_shape[1]):
#                 self.models[y, x] = model.CTS(context_length=context_length, alphabet=alphabet)

#         self.context_functor = context_functor

#     def update(self, frame):
#         total_log_probability = 0.0
#         for y in range(frame.shape[0]):
#             for x in range(frame.shape[1]):
#                 context = self.context_functor(frame, y, x)
#                 colour = frame[y, x]
#                 total_log_probability += self.models[y, x].update(context=context, symbol=colour)
#                 #total_log_probability_after += self.models[y, x].log_prob(context=context, symbol=colour)
#         return total_log_probability

#     def query(self, frame):
#         total_log_probability = 0.0
#         for y in range(frame.shape[0]):
#             for x in range(frame.shape[1]):
#                 context = self.context_functor(frame, y, x)
#                 colour = frame[y, x]
#                 total_log_probability += self.models[y, x].log_prob(context=context, symbol=colour)
#         return total_log_probability

#     def sample(self):
#         output_frame = np.zeros((self.frame_shape[0], self.frame_shape[1]), dtype=np.uint32)
#         for y in range(self.frame_shape[0]):
#             for x in range(self.frame_shape[1]):
#                 # From a programmer's perspective, this is why we must respect the chain rule: otherwise
#                 # we condition on garbage.
#                 context = self.context_functor(output_frame, y, x)
#                 output_frame[y, x] = self.models[y, x].sample(context=context, rejection_sampling=True)

#         return output_frame
#version without symbol frame
class ConvolutionalDensityModel(object):
    """A density model for Freeway frames.
    
    This one predict according to an L-shaped context around the current pixel.
    """
    def __init__(self, frame_shape, context_functor, alphabet=None):
        """Constructor.
        
        Args:
            init_frame: A sample frame (numpy array) from which we determine the shape and type of our data.
            context_functor: Function mapping image x position to a context.
        """
        # For efficiency, we'll pre-process the frame into our internal representation.
        self.frame_shape = frame_shape
        #self.symbol_frame = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.uint32)

        context_length = len(context_functor(np.zeros((frame_shape[0:2]), dtype=np.uint32), -1, -1))
        self.convolutional_model = model.CTS(context_length=context_length, alphabet=alphabet)
        
        self.context_functor = context_functor
    def update(self, frame):
        
        total_log_probability = 0.0
        for y in range(frame.shape[0]):
            for x in range(frame.shape[1]):
                context = self.context_functor(frame, y, x)
                colour = frame[y, x]
                total_log_probability += self.convolutional_model.update(context=context, symbol=colour)

        return total_log_probability
    def query(self, frame):

        total_log_probability = 0.0
        for y in range(frame.shape[0]):
            for x in range(frame.shape[1]):
                context = self.context_functor(frame, y, x)
                colour = frame[y, x]
                total_log_probability += self.convolutional_model.log_prob(context=context, symbol=colour)
        return total_log_probability

    def sample(self):
        output_frame = np.zeros((*self.frame_shape, 3), dtype=np.uint8)
        #print (output_frame.shape)
        
        for y in range(self.frame_shape[0]):
            for x in range(self.frame_shape[1]):
                # From a programmer's perspective, this is why we must respect the chain rule: otherwise
                # we condition on garbage.
                context = self.context_functor(output_frame, y, x)
                output_frame[y, x] = self.convolutional_model.sample(context=context,
                                                                          rejection_sampling=True)

        return output_frame

    def sample_with_update(self):
        output_frame = np.zeros((*self.symbol_frame.shape, 3), dtype=np.uint8)
        #print (output_frame.shape)
        
        for y in range(self.symbol_frame.shape[0]):
            for x in range(self.symbol_frame.shape[1]):
                context = self.context_functor(self.symbol_frame, y, x)
                self.symbol_frame[y, x] = self.convolutional_model.sample(context=context,
                                                                          rejection_sampling=False)
                self.convolutional_model.update(context, self.symbol_frame[y, x])
    
        return symbols_to_rgb(self.symbol_frame, output_frame)