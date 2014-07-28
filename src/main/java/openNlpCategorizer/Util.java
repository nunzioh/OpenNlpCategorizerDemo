package openNlpCategorizer;

import java.io.IOException;
import java.io.InputStream;
import java.io.UnsupportedEncodingException;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import opennlp.tools.doccat.DoccatModel;
import opennlp.tools.doccat.DocumentCategorizerME;
import opennlp.tools.doccat.DocumentSample;
import opennlp.tools.doccat.DocumentSampleStream;
import opennlp.tools.doccat.FeatureGenerator;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.PlainTextByLineStream;

/**
 * Some helper methods for working with the OpenNlp Categorizer
 *
 */
public class Util {
	
	private static InputStream fromResourceFile(String trainFileResource) throws UnsupportedEncodingException {
		InputStream dataIn = Thread.currentThread().getContextClassLoader().getResourceAsStream(trainFileResource);
		return dataIn; 
	}
	
	private static ObjectStream<DocumentSample> lineDocSampler(InputStream dataIn) throws UnsupportedEncodingException {
		ObjectStream<String> lineStream =
				new PlainTextByLineStream(dataIn, "UTF-8");
		ObjectStream<DocumentSample> sampleStream = new DocumentSampleStream(lineStream);
		return sampleStream; 
	}
	
	/**
	 * Train a new categorizer model using trainFile and fg
	 * @param trainFileResource - filename of a file from classpath
	 * @param cutoff - number of times a feature needs to appear to be included in model 
	 * @param fg
	 * @return
	 */
	public static DoccatModel customTrainResource(String trainFileResource, int cutoff, FeatureGenerator... fg) {
		DoccatModel model = null;
		InputStream dataIn = null;
		try {
			dataIn = fromResourceFile(trainFileResource);
			model = DocumentCategorizerME.train("en", lineDocSampler(dataIn), cutoff, 100, fg);
		} catch (IOException e) {
			// Failed to read or parse training data, training failed
			e.printStackTrace();
		} finally {
			if (dataIn != null) {
				try {
					dataIn.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
		return model; 
	}
	
	public static DocumentCategorizerME customCategorizer(DoccatModel m, FeatureGenerator... fg) {
		return new DocumentCategorizerME(m, fg);
	}
 	
	public static String bestOf(Map<String, Double> results) {
		String best = null; 
		double bestValue = 0.0; 
		for (Entry<String, Double> entry: results.entrySet()) {
			double val = entry.getValue();
			if (val > bestValue) {
				bestValue = val; 
				best = entry.getKey(); 
			}
		}
		return best;
	}
	
	/**
	 * Categorize the inputText and return a map of the results
	 * @param cat
	 * @param inputText
	 * @param fg
	 * @return all results from categorization
	 */
	public static Map<String, Double> test(DocumentCategorizerME cat, String inputText, FeatureGenerator... fg) {
		Map<String, Double> results = new HashMap<String,Double>(); 
		double[] outcomes = cat.categorize(inputText);
		for (int i=0; i < cat.getNumberOfCategories(); i++) {
			results.put(cat.getCategory(i), outcomes[i]); 
		}
		return results; 
	}
	
	/**
	 * Categorize the inputText and return the best category 
	 * @param cat
	 * @param inputText
	 * @param fg
	 * @return
	 */
	public static String testAndGetBest(DocumentCategorizerME cat, String inputText, FeatureGenerator... fg) {
		double[] outcomes = cat.categorize(inputText);
		String best = cat.getBestCategory(outcomes);
		return best; 
	}
	
	public static String testAndSummarizeAll(DocumentCategorizerME cat, String inputText, FeatureGenerator... fg) {
		double[] outcomes = cat.categorize(inputText);
		String all = cat.getAllResults(outcomes);
		return all; 
	}
}
