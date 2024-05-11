class simplex:
    def relationship(self, points:dict):
        temp = []
        for _,v in points.items():
            temp.append(v)
        temp.sort()
        for k,v in points.items():
            if v == temp[0]:
                self.b = k
            elif v == temp[1]:
                self.g = k
            elif v == temp[2]:
                self.w = k
        print(f'B: {self.b}\nG: {self.g}\nW: {self.w}')
    
    def calc_m(self):
        x = (self.g[0] + self.b[0])/2
        y = (self.g[1] + self.b[1])/2
        self.m = (x,y)

    def reflection(self):
        x = (self.m[0] - self.w[0])
        y = (self.m[1] - self.w[1])
        rx = self.m[0] + x
        ry = self.m[1] + y
        self.r = (rx,ry)
        print(f'R: {self.r}')


    def expansion(self):
        x = (self.m[0] - self.w[0])*2
        y = (self.m[1] - self.w[1])*2
        ex = self.m[0] + x
        ey = self.m[1] + y
        self.e = (ex,ey)
        print(f'E: {self.e}')
    def contraction(self):
        x = (self.m[0] - self.w[0])*0.5
        y = (self.m[1] - self.w[1])*0.5
        cx = self.w[0] + x
        cy = self.w[1] + y
        self.c = (cx,cy)
        print(f'C: {self.c}')
    def shrink(self):
        x = (self.w[0] + self.b[0])/2
        y = (self.w[1] + self.b[1])/2
        self.s = (x,y)
        print(f'S: {self.s}')
