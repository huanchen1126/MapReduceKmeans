import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.RunningJob;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.hadoop.mapred.TextOutputFormat;

public class Kmeans {
  public static String CENTROID_PATH = "CENTROID_FILE";

  public static String NUM_K = "NUM_K";

  public static String NUM_FEATURE = "NUM_FEATURE";

  public static class KmeansMap extends MapReduceBase implements
          Mapper<LongWritable, Text, Text, Text> {
    public Centroid[] centroids;

    public int k;

    public void configure(JobConf job) {
      Configuration conf = new Configuration();
      /* initialize the centroid */
      this.k = Integer.parseInt(job.get(Kmeans.NUM_K));
      centroids = new Centroid[this.k];
      BufferedReader br = null;
      try {
        Path path = new Path(job.get(Kmeans.CENTROID_PATH));
        FileSystem fs = path.getFileSystem(conf);
        FileStatus[] fileStatus = fs.listStatus(path);
        for (FileStatus fstatus : fileStatus) {
          Path p = fstatus.getPath();
          if (p.toString().contains("_SUCCESS"))
            continue;
          FSDataInputStream fsdi = fs.open(p);
          br = new BufferedReader(new InputStreamReader(fsdi));
          /* the file structure is clusterID,centroid,feature_vector */
          /* read the centroid file to centroids */
          while (true) {
            String line = br.readLine();
            if (line == null)
              break;
            int indStart = line.indexOf(',');
            int indEnd = line.indexOf(',', indStart + 1);
            String clusterid = line.substring(0, indStart);
            String feature = line.substring(indEnd + 1);
            centroids[Integer.parseInt(clusterid) - 1] = new Centroid(clusterid, feature);
          }
        }
      } catch (IOException e) {
        e.printStackTrace();
      } finally {
        if (br != null)
          try {
            br.close();
          } catch (IOException e) {
            e.printStackTrace();
          }
      }
    }

    public void map(LongWritable key, Text value, OutputCollector<Text, Text> output,
            Reporter reporter) throws IOException {
      String line = value.toString();
      if (line.startsWith("c"))
        return;
      int indStart = line.indexOf(',');
      String feature = line.substring(indStart + 1);
      String[] features = feature.split(",");
      int cluster = 0;
      double similarity = computeSimilarity(centroids[0].features, features);

      for (int i = 1; i < k; i++) {
        double sim = computeSimilarity(centroids[i].features, features);
        if (sim < similarity) {
          cluster = i;
          similarity = sim;
        }
      }
      output.collect(new Text(String.valueOf(centroids[cluster].id)), new Text(feature));
    }

    private double computeSimilarity(String[] features1, String[] features2) {
      double result = 0;
      for (int i = 0; i < features1.length; i++) {
        double f1 = Float.parseFloat(features1[i]);
        double f2 = Float.parseFloat(features2[i]);
        double abs = Math.abs(f1 - f2);
        result += abs * abs;
      }
      return result;
    }

  }

  /**
   * The calss for centroid
   * 
   * @author huanchen
   * 
   */
  public static class Centroid {
    public String id;

    public String[] features;

    public Centroid(String id, String feature) {
      this.id = id;
      this.features = feature.split(",");
    }
  }

  public static class KmeansCombiner extends MapReduceBase implements
          Reducer<Text, Text, Text, Text> {
    public void reduce(Text key, Iterator<Text> values, OutputCollector<Text, Text> output,
            Reporter reporter) throws IOException {
      /* number of samples in this cluster */
      long count = 1;
      /* first sample features */
      String[] features = values.next().toString().split(",");
      /* initialize new centroid */
      double[] centroid = new double[features.length];
      for (int i = 0; i < features.length; i++) {
        centroid[i] = Float.parseFloat(features[i]);
      }
      while (values.hasNext()) {
        count++;
        features = values.next().toString().split(",");
        /* update centroid */
        for (int i = 0; i < features.length; i++) {
          centroid[i] += Float.parseFloat(features[i]);
        }
      }
      /* transform centroid to string */
      String cent = "";
      for (int i = 0; i < centroid.length; i++) {
        cent += "," + String.valueOf(centroid[i]);
      }
      /* output the centroid */
      output.collect(key, new Text(count + cent));
    }
  }

  public static class KmeansReduce extends MapReduceBase implements Reducer<Text, Text, Text, Text> {
    public Centroid[] centroids;

    public int k;

    public static double diff = 0.0001;

    public void configure(JobConf job) {
      Configuration conf = new Configuration();
      /* initialize the centroid */
      this.k = Integer.parseInt(job.get(Kmeans.NUM_K));
      centroids = new Centroid[this.k];
      BufferedReader br = null;
      try {
        Path path = new Path(job.get(Kmeans.CENTROID_PATH));
        FileSystem fs = path.getFileSystem(conf);
        FileStatus[] fileStatus = fs.listStatus(path);
        for (FileStatus fstatus : fileStatus) {
          Path p = fstatus.getPath();
          if (p.toString().contains("_SUCCESS"))
            continue;
          FSDataInputStream fsdi = fs.open(p);
          br = new BufferedReader(new InputStreamReader(fsdi));
          /* the file structure is clusterID,centroid,feature_vector */
          /* read the centroid file to centroids */
          while (true) {
            String line = br.readLine();
            if (line == null)
              break;
            int indStart = line.indexOf(',');
            int indEnd = line.indexOf(',', indStart + 1);
            String clusterid = line.substring(0, indStart);
            String feature = line.substring(indEnd + 1);
            centroids[Integer.parseInt(clusterid) - 1] = new Centroid(clusterid, feature);
          }
        }
      } catch (IOException e) {
        e.printStackTrace();
      } finally {
        if (br != null)
          try {
            br.close();
          } catch (IOException e) {
            e.printStackTrace();
          }
      }
    }

    public void reduce(Text key, Iterator<Text> values, OutputCollector<Text, Text> output,
            Reporter reporter) throws IOException {
      long count = 0;
      double[] centroid = null;
      while (values.hasNext()) {
        String line = values.next().toString();
        int ind = line.indexOf(',');
        count += Long.parseLong(line.substring(0, ind));
        String[] features = line.substring(ind + 1).split(",");
        if (centroid == null)
          centroid = new double[features.length];
        addToCentroid(centroid, features);
      }
      /* if no case falls into this cluster, assume the previous centroid is one such case */
      if (centroid == null) {
        /* get the centroid of the same cluster id in previous iteration */
        String[] prevCentroidFeatures = centroids[Integer.parseInt(key.toString()) - 1].features;
        centroid = new double[prevCentroidFeatures.length];
        addToCentroid(centroid, prevCentroidFeatures);
        count = 1;
      }
      /* average to get the new centroid */
      /* this is a fake num to make the file structure consistent to initial centroid */
      String cent = ",0";
      for (int i = 0; i < centroid.length; i++) {
        centroid[i] /= count;
        cent += "," + centroid[i];
      }
      /* get the centroid of the same cluster id in previous iteration */
      String[] prevCentroidFeatures = centroids[Integer.parseInt(key.toString()) - 1].features;
      /* detect if this cluster is converged */
      boolean notConverged = false;
      for (int i = 0; i < prevCentroidFeatures.length; i++) {
        double prevCFeature = Double.parseDouble(prevCentroidFeatures[i]);
        double curCFeature = centroid[i];
        if (Math.abs(prevCFeature - curCFeature) > this.diff) {
          notConverged = true;
          break;
        }
      }
      if (!notConverged)
        reporter.getCounter(Kmeans.KmeansReduce.Counter.CONVERGED).increment(1);
      output.collect(null, new Text(key.toString() + cent));
    }

    /**
     * add features to centroid
     * 
     * @param centroid
     * @param features
     */
    public void addToCentroid(double[] centroid, String[] features) {
      if (centroid == null || features == null)
        return;
      for (int i = 0; i < features.length; i++) {
        centroid[i] += Double.parseDouble(features[i]);
      }
    }

    /**
     * the counter for detecting convergence
     */
    public static enum Counter {
      CONVERGED(0);
      long counter;

      private Counter(long c) {
        counter = c;
      }

      public void increment(long c) {
        this.counter += c;
      }

      public long getCounter() {
        return this.counter;
      }
    }
  }

  public static void main(String[] args) throws IOException {
    /* k should be at least 2 */
    if (args.length != 4)
      return;
    String inputPath = args[0];
    String outputPath = args[1];
    String centroidPath = args[2];
    String numK = args[3];
    int k = Integer.parseInt(numK);
    long counter = 0;
    long iteration = 1;
    while (counter < k) {
      System.out.println("-------Iteration " + iteration + " --------");
      String curoutputPath = outputPath + "/centroid_iteration_" + iteration;
      if (iteration != 1) {
        centroidPath = outputPath + "/centroid_iteration_" + (iteration - 1);
      }
      JobConf job = new JobConf(Kmeans.class);
      job.setJobName("K means");
      job.setOutputKeyClass(Text.class);
      job.setOutputValueClass(Text.class);
      job.setMapperClass(KmeansMap.class);
      job.setCombinerClass(KmeansCombiner.class);
      job.setReducerClass(KmeansReduce.class);
      job.setInputFormat(TextInputFormat.class);
      job.setOutputFormat(TextOutputFormat.class);
      job.set(Kmeans.CENTROID_PATH, centroidPath);
      job.set(Kmeans.NUM_K, numK);
      FileInputFormat.setInputPaths(job, new Path(inputPath));
      FileOutputFormat.setOutputPath(job, new Path(curoutputPath));
      RunningJob runningJob = JobClient.runJob(job);
      counter = runningJob.getCounters().findCounter(Kmeans.KmeansReduce.Counter.CONVERGED)
              .getCounter();
      System.out.println("-------Iteration " + iteration + " Counter " + counter + " --------");
      iteration++;
    }
  }
}
