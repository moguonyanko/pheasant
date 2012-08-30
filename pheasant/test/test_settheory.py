#!/usr/bin/python3
# -*- coding: utf-8 -*-

import math
import unittest

import pheasant.settheory as se

class TestEventBehavior(unittest.TestCase):
    '''
    Test of event behavior in probability theory.
    '''
    def test_exclusivep(self):
        '''
        Test check to be exclusive function.
        '''
        s1 = {1,2,3,4,5}
        s2 = {6,7,8,9,10}
        res = se.exclusivep(s1, s2)
        self.assertEqual(True, res)
        s3 = {1,2,10,4,5}
        s4 = {5,7,8,9,10}
        res = se.exclusivep(s3, s4)
        self.assertEqual(False, res)

        evt1 = se.Event(s1)
        evt2 = se.Event(s2)
        self.assertEqual(True, evt1.exclusivep(evt2))

    def test_eventadd(self):
        '''
        Event addtion test.
        '''
        s1 = se.Event({1,2})
        s2 = se.Event({3,4})
        res = s1+s2
        self.assertEqual(se.Event({1,2,3,4}), res)

    def test_eventmul(self):
        '''
        Event addtion test.
        '''
        s1 = se.Event({1,2,3,4})
        s2 = se.Event({1,4})
        res = s1*s2
        self.assertEqual(se.Event({1,4}), res)

    def test_eventmod(self):
        '''
        Event modulo test.
        '''
        s1 = se.Event({1,2,3,4})
        s2 = se.Event({1,4})
        res = s1%s2
        self.assertEqual(se.Event({2,3}), res)
        res = s2%s1
        self.assertEqual(se.Event({}), res)

    def test_emptyp(self):
        '''
        Test empty check function.
        '''
        s1 = se.Event({})
        self.assertEqual(True, s1.emptyp())
        s2 = se.Event({1})
        self.assertEqual(False, s2.emptyp())

    def test_injectp(self):
        '''
        Whether events injection.
        '''
        pass

    def test_surjectp(self):
        '''
        Whether events surjection.
        '''
        pass

    def test_bijectp(self):
        '''
        Whether events bijection.
        '''
        pass

if __name__ == '__main__':
	print(__file__)
	unittest.main()

