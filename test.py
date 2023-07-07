#!/usr/bin/env python3

class test:
    def __init__(self, x) -> None:
        self.x = x
    def testprint(self):
        print(self.x.x)

class test2:
    def __init__(self,x) -> None:
        self.x = x

Test2 = test2(2)
Test = test(Test2)
Test2.x = 4
Test.testprint()