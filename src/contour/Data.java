package contour;

/** Seeded synthetic two-moons data. Train and test are independent draws. */
public final class Data {
    private Data() {}
    public static final class Random {
        private int state;
        public Random(int seed){state=seed;}
        public double next(){state=1664525*state+1013904223;return Integer.toUnsignedLong(state)/4294967296.0;}
    }
    public static double[][] moons(int count,int seed) {
        if(count%2!=0||count<2)throw new IllegalArgumentException("Even balanced sample count required");
        var r=new Random(seed);double[][] rows=new double[count][3];
        for(int i=0;i<count;i++){
            int label=i%2;double angle=r.next()*Math.PI;
            double x=label==0?Math.cos(angle):1-Math.cos(angle);
            double y=label==0?Math.sin(angle):.5-Math.sin(angle);
            // Irwin-Hall noise: centered sum of six uniforms, std approx .10.
            double nx=0,ny=0;for(int j=0;j<6;j++){nx+=r.next()-.5;ny+=r.next()-.5;}
            rows[i]=new double[]{x+nx*.14-.5,y+ny*.14-.25,label};
        }
        return rows;
    }
}
