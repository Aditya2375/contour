package contour;
import java.util.Arrays;
public final class RefDump {
    public static void main(String[] args){
        var train=Data.moons(128,17);var test=Data.moons(256,91);var model=new Network(42);var optimizer=new Network.Adam();
        StringBuilder out=new StringBuilder("[");
        for(int target:new int[]{0,1,10,100,600}){
            while(optimizer.steps<target)optimizer.step(model,train,.02);
            if(target>0)out.append(',');
            double[] gradient=new double[Network.COUNT];double value=model.loss(train,gradient);
            out.append("{\"step\":").append(target).append(",\"weights\":").append(Arrays.toString(model.weights))
               .append(",\"gradient\":").append(Arrays.toString(gradient)).append(",\"loss\":").append(value)
               .append(",\"testLoss\":").append(model.loss(test,null)).append(",\"accuracy\":").append(model.accuracy(test)).append('}');
        }
        System.out.println(out.append(']'));
    }
}
