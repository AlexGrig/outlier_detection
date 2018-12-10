#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# THIS FILE IS COPY-PASTED FROM HERE: https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/04-utils/tensorboard

# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514

"""
This Logger logs numpy variables, not tensorflow graph nodes. 
"""  

import tensorflow as tf
import numpy as np
import scipy.misc
import os, shutil # for cleaning up the folder

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x


class Logger(object):
    def __init__(self, log_dir, graph=None):
        """Create a summary writer logging to log_dir."""
        self.summary_folder = log_dir
        if graph is None:
            self.writer = tf.summary.FileWriter(log_dir)
        else:
            self.writer = tf.summary.FileWriter(log_dir, graph)
            
    def clean_log_folder(self,p_clean=True):
        """
        Cleans the log folder.
        
        Inputs:
        -------------------------
        
        p_clean: bool
            If true clean the folder if false do nothing.
        """
        #import pdb; pdb.set_trace()
        path = os.path.abspath(self.summary_folder)
        if p_clean:
            for the_file in os.listdir(path):
                file_path = os.path.join(path, the_file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path): 
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(e)
    
    def graph_summary(self, graph, step):
        """
        """
        self.writer.add_graph(graph, global_step = step)
        self.writer.flush()
                 
    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)
        self.writer.flush()
        
    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
          # Write the image to a string
          try:
            s = StringIO()
          except:
            s = BytesIO()
          scipy.misc.toimage(img).save(s, format="png")
    
          # Create an Image object
          img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                     height=img.shape[0],
                                     width=img.shape[1])
          # Create a Summary value
          img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))
    
        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)
        self.writer.flush()
        
    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)
    
        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))
    
        # Drop the start of the first bin
        bin_edges = bin_edges[1:]
    
        # Add bin edges and counts
        for edge in bin_edges:
          hist.bucket_limit.append(edge)
        for c in counts:
          hist.bucket.append(c)
    
        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()
    
def logger_test():    
    
    ll = Logger('./logger_test')
    ll.clean_log_folder(True)
    
    for ii in range(0,10):
        ll.scalar_summary('test', (ii+2)**2,ii)
        print(ii+5, ii)
        
if __name__ == "__main__":
    logger_test()