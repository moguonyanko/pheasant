#!/usr/bin/python3
# -*- coding: utf-8 -*-

import math

class Event():
    '''
    This class is to express event in probability theory.
    '''
    def __init__(self, elements):
        '''
        This initializer recieve elements set.
        '''
        self.elements = elements

    def __add__(self, target):
        '''
        Return addition event.
        '''
        return Event(self.elements.union(target.elements))

    def __mul__(self, target):
        '''
        Return multiplication event.
        '''
        return Event(self.elements.intersection(target.elements))

    def __mod__(self, target):
        '''
        Return modulo event.
        Left side is to consider as all event.
        '''
        return Event(self.elements.difference(target.elements))

    def __eq__(self, target):
        '''
        If elements are equal, events are equal.
        '''
        if len(self.elements) == 0 and len(target.elements) == 0:
            return True
        else:
            return self.elements == target.elements

    def __str__(self):
        '''
        Return string expression of self elements.
        '''
        return str(list(self.elements))

    def exclusivep(self, target):
        '''
        Is self elements and target elements exclusive?
        '''
        return exclusivep(self.elements, target.elements)

    def emptyp(self):
        '''
        Is this event empty event?
        '''
        return len(self.elements) == 0

    def injectp(self, target):
        '''
        Is self and target injection?
        '''
        pass

    def surjectp(self, target):
        '''
        Is self and target surjection?
        '''
        pass

    def bijectp(self, target):
        '''
        Is self and target bijection?
        '''
        pass

def exclusivep(s1, s2):
    '''
    Is s1 and s2 exclusive?
    '''
    interset = s1.intersection(s2)
    return len(interset) == 0

if __name__ == '__main__':
	print("settheory module load")
