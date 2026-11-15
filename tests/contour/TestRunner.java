package contour;
public final class TestRunner {
    static int checks;
    static void check(String s,boolean b){if(!b)throw new AssertionError(s);checks++;System.out.println("ok "+s);}
    public static void main(String[] args){
        var data=Data.moons(128,17);var test=Data.moons(256,91);var model=new Network(42);
        double[] g=new double[Network.COUNT];model.loss(data,g);double worst=0;
        for(int i=0;i<g.length;i++){
            double old=model.weights[i],eps=1e-5;model.weights[i]=old+eps;double plus=model.loss(data,null);model.weights[i]=old-eps;double minus=model.loss(data,null);model.weights[i]=old;
            worst=Math.max(worst,Math.abs((plus-minus)/(2*eps)-g[i]));
        }
        check("all parameter gradients match finite differences",worst<1e-8);System.out.println("gradient max error "+worst);
        check("parameter count is 105",Network.COUNT==105);
        check("sigmoid positive extreme stable",Network.sigmoid(1000)==1);
        check("sigmoid negative extreme stable",Network.sigmoid(-1000)==0);
        var repeat=new Network(42);check("deterministic initialization",java.util.Arrays.equals(model.weights,repeat.weights));
        var zero=new Network(42);java.util.Arrays.fill(zero.weights,0);
        check("zero weights yield zero logit",zero.logit(2,-7)==0);
        check("zero logit BCE equals ln2",Math.abs(zero.loss(new double[][]{{2,-7,1}},g)-Math.log(2))<1e-15);
        check("single positive target output-bias gradient is minus half",g[104]==-.5);
        check("all other zero-network derivatives zero",java.util.Arrays.stream(java.util.Arrays.copyOf(g,104)).allMatch(x->x==0));
        var first=new Network.Adam();first.step(zero,new double[][]{{2,-7,1}},.02);
        check("Adam first step analytic update",Math.abs(zero.weights[104]-.02*.5/(.5+1e-8))<1e-15);
        zero.weights[104]=1000;
        check("extreme wrong-positive loss finite and correct",zero.loss(new double[][]{{0,0,0}},null)==1000);
        zero.weights[104]=-1000;
        check("extreme wrong-negative loss finite and correct",zero.loss(new double[][]{{0,0,1}},null)==1000);
        check("balanced labels",java.util.Arrays.stream(data).filter(r->r[2]==1).count()==64);
        check("independent test draw",data[0][0]!=test[0][0]);
        for(int seed:new int[]{42,7,101})for(double rate:new double[]{.005,.02,.05}) {
            var m=new Network(seed);var opt=new Network.Adam();double before=m.loss(data,null);
            for(int i=0;i<600;i++)opt.step(m,data,rate);
            check("finite decreasing loss seed "+seed+" rate "+rate,Double.isFinite(m.loss(data,null))&&m.loss(data,null)<before*.5);
        }
        double start=model.loss(data,null);var adam=new Network.Adam();
        for(int i=0;i<600;i++)adam.step(model,data,.02);
        double end=model.loss(data,null),accuracy=model.accuracy(test);
        check("training loss falls",end<start*.1);
        check("held-out balanced accuracy exceeds 90 percent",accuracy>.9);
        check("all weights remain finite",java.util.Arrays.stream(model.weights).allMatch(Double::isFinite));
        System.out.printf("loss %.8f -> %.8f, heldout accuracy %.6f%n",start,end,accuracy);
        System.out.println(checks+" checks passed");
    }
}
