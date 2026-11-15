package contour;
import java.util.Arrays;

/** Two-input tanh MLP, stable binary cross-entropy, hand-derived reverse pass. */
public final class Network {
    public static final int WIDTH=8;
    public static final int COUNT=2*WIDTH+WIDTH+WIDTH*WIDTH+WIDTH+WIDTH+1;
    public final double[] weights=new double[COUNT];
    public Network(int seed) {
        Data.Random random=new Data.Random(seed);
        // Xavier uniform for each dense weight matrix; biases start at zero.
        for(int j=0;j<WIDTH;j++)for(int i=0;i<2;i++)weights[j*2+i]=(random.next()*2-1)*Math.sqrt(6.0/(2+WIDTH));
        int second=3*WIDTH;
        for(int j=0;j<WIDTH;j++)for(int i=0;i<WIDTH;i++)weights[second+j*WIDTH+i]=(random.next()*2-1)*Math.sqrt(6.0/(2*WIDTH));
        int output=second+WIDTH*WIDTH+WIDTH;
        for(int i=0;i<WIDTH;i++)weights[output+i]=(random.next()*2-1)*Math.sqrt(6.0/(WIDTH+1));
    }
    public double logit(double x,double y) { return forward(x,y,new double[WIDTH],new double[WIDTH]); }
    private double forward(double x,double y,double[] a,double[] b) {
        int second=3*WIDTH, bias=second+WIDTH*WIDTH,output=bias+WIDTH;
        for(int j=0;j<WIDTH;j++)a[j]=Math.tanh(weights[2*j]*x+weights[2*j+1]*y+weights[2*WIDTH+j]);
        for(int j=0;j<WIDTH;j++) {double z=weights[bias+j];for(int i=0;i<WIDTH;i++)z+=weights[second+j*WIDTH+i]*a[i];b[j]=Math.tanh(z);}
        double z=weights[output+WIDTH];for(int i=0;i<WIDTH;i++)z+=weights[output+i]*b[i];return z;
    }
    public static double sigmoid(double z) {return z>=0?1/(1+Math.exp(-z)):Math.exp(z)/(1+Math.exp(z));}
    public double loss(double[][] data,double[] gradient) {
        if(gradient!=null)Arrays.fill(gradient,0);
        double total=0;
        int second=3*WIDTH,bias=second+WIDTH*WIDTH,output=bias+WIDTH;
        double[] a=new double[WIDTH],b=new double[WIDTH],da=new double[WIDTH];
        for(double[] row:data) {
            double z=forward(row[0],row[1],a,b),label=row[2];
            total+=Math.max(z,0)-z*label+Math.log1p(Math.exp(-Math.abs(z)));
            if(gradient==null)continue;
            double dz=(sigmoid(z)-label)/data.length;
            gradient[output+WIDTH]+=dz;Arrays.fill(da,0);
            for(int j=0;j<WIDTH;j++) {
                gradient[output+j]+=dz*b[j];
                double db=dz*weights[output+j]*(1-b[j]*b[j]);gradient[bias+j]+=db;
                for(int i=0;i<WIDTH;i++){gradient[second+j*WIDTH+i]+=db*a[i];da[i]+=db*weights[second+j*WIDTH+i];}
            }
            for(int j=0;j<WIDTH;j++) {
                double u=da[j]*(1-a[j]*a[j]);gradient[2*j]+=u*row[0];gradient[2*j+1]+=u*row[1];gradient[2*WIDTH+j]+=u;
            }
        }
        return total/data.length;
    }
    public double accuracy(double[][] data) {int correct=0;for(double[] row:data)if((logit(row[0],row[1])>=0?1:0)==(int)row[2])correct++;return (double)correct/data.length;}
    public static final class Adam {
        private final double[] m=new double[COUNT],v=new double[COUNT],gradient=new double[COUNT];
        public int steps;
        public void step(Network model,double[][] data,double rate) {
            model.loss(data,gradient);steps++;
            for(int i=0;i<COUNT;i++) {
                m[i]=.9*m[i]+.1*gradient[i];v[i]=.999*v[i]+.001*gradient[i]*gradient[i];
                model.weights[i]-=rate*(m[i]/(1-Math.pow(.9,steps)))/(Math.sqrt(v[i]/(1-Math.pow(.999,steps)))+1e-8);
            }
        }
    }
}
