package openNlpCategorizer;

import java.util.Map;

import opennlp.model.IndexHashTable;
import opennlp.model.MutableContext;
import opennlp.tools.doccat.BagOfWordsFeatureGenerator;
import opennlp.tools.doccat.DoccatModel;
import opennlp.tools.doccat.DocumentCategorizerME;
import opennlp.tools.doccat.FeatureGenerator;

import org.junit.Assert;
import org.junit.Test;

/**
 * A few demo tests exercising the OpenNlp categorizer with custom feature generation
 * @author nunzioh
 *
 */
public class CategorizerDemo extends Assert {
	
	// Labels
	private final static String COFFEE = "COFFEE";
	private final static String BEER = "BEER";

	// Training file with COFFEE and BEER reviews
	private String TRAIN_FILE = "coffee.train";

	private String TEST_COFFEE = "I LOVE THIS! I thought that I preferred the lighter roasted stuff... and then I got this one in the mail. I could smell the difference before I even brewed it. I'm a dark roast kinda guy now! ";
	private String TEST_BEER = "Smells the classic lager smell but with the sweet, malty smell. Taste has that crisp, lager taste with the sweetness of the malts coming in. ";

	// "In sample" meaning it was taken from one of the coffee reviews in the training file
	private String IN_SAMPLE_TEST_COFFEE = "I just ordered 2 more cans of this.";

	/**
	 * Generate both uni-gram (BagOfWords) and bi-gram features
	 */
	private FeatureGenerator[] uniAndBigramFeatureGen = new FeatureGenerator[] {
			new BagOfWordsFeatureGenerator(), new BigramFeatureGenerator() };

	/**
	 * Test that the trained model actually contains the features we expect
	 */
	@Test
	public void testModel() {
		FeatureGenerator[] fg = uniAndBigramFeatureGen;
		DoccatModel cm = Util.customTrainResource(TRAIN_FILE, 0, fg);

		Object[] ds = cm.getChunkerModel().getDataStructures();

		String[] outcomeNames = (String[]) ds[2];
		System.out.println(outcomeNames);

		IndexHashTable<String> featureTable = (IndexHashTable<String>) ds[1];
		MutableContext[] mc = (MutableContext[]) ds[0];

		int coffeeIndex = featureTable.get("bow=coffee");

		// make sure there is a unigram coffee feature
		assertNotEquals(-1, coffeeIndex);

		int[] coffeeOutcomeIndexes = mc[coffeeIndex].getOutcomes();

		// coffee should only appear under one outcome (the coffee one)
		assertEquals(1, coffeeOutcomeIndexes.length);

		// make sure coffee is in the COFFEE outcome
		assertEquals("COFFEE", outcomeNames[coffeeOutcomeIndexes[0]]);

		// test beer unigram
		int beerIndex = featureTable.get("bow=beer");
		assertNotEquals(-1, beerIndex);
		int[] beerOutcomeIndexes = mc[beerIndex].getOutcomes();
		assertEquals(1, beerOutcomeIndexes.length);
		assertEquals("BEER", outcomeNames[beerOutcomeIndexes[0]]);

		// test bigram medium roast
		int mrIndex = featureTable.get("bg=medium:roast");
		int[] mrOutcomeIndexes = mc[mrIndex].getOutcomes();
		assertEquals(1, mrOutcomeIndexes.length);
		assertEquals("COFFEE", outcomeNames[mrOutcomeIndexes[0]]);

		// test feature that has two categories
		int mouthfeelIndex = featureTable.get("bow=mouthfeel");
		int[] mouthfeelOutcomeIndexes = mc[mouthfeelIndex].getOutcomes();
		assertEquals(2, mouthfeelOutcomeIndexes.length);
	}

	@Test
	public void test() {
		FeatureGenerator[] fg = uniAndBigramFeatureGen;

		DoccatModel model = Util
				.customTrainResource(TRAIN_FILE, 0, fg);
		assertNotNull(model);

		DocumentCategorizerME cat = Util.customCategorizer(model, fg);

		assertEquals(COFFEE, Util.testAndGetBest(cat, "coffee", fg));
		assertEquals(COFFEE, Util.testAndGetBest(cat, "of coffee", fg));
		assertEquals(COFFEE, Util.testAndGetBest(cat, "medium roast of coffee blah of_coffee", fg));
		assertEquals(COFFEE, Util.testAndGetBest(cat, IN_SAMPLE_TEST_COFFEE, fg));
	
		// "cans" appears twice under COFFEE, but only once for BEER
		assertEquals(COFFEE, Util.testAndGetBest(cat, "cans", fg));
		
		assertEquals(BEER, Util.testAndGetBest(cat, "beer", fg));
		assertEquals(BEER, Util.testAndGetBest(cat, "juicy grapefruit", fg));
	}

	private void testAllModels(String inputTest, String expected,
			DocumentCategorizerME catUni, DocumentCategorizerME catBigram,
			DocumentCategorizerME catUniAndBigram) {
		System.out.println("Test: " + inputTest);

		Map<String, Double> resultsUni = Util.test(catUni, inputTest);
		Map<String, Double> resultsUniAndBi = Util.test(catUniAndBigram,
				inputTest);
		Map<String, Double> resultsBi = Util.test(catBigram, inputTest);

		System.out.println("UniAndBigram model: " + resultsUniAndBi);
		System.out.println("Bigram model: " + resultsBi);
		System.out.println("Uni model: " + resultsUni);

		assertEquals("UniAndBigram model test failed", expected,
				Util.bestOf(resultsUniAndBi));
		assertEquals("Bigram model test failed", expected,
				Util.bestOf(resultsBi));
		assertEquals("Unigram model test failed", expected,
				Util.bestOf(resultsUni));
	}

	@Test
	public void testCompare() {
		FeatureGenerator[] fgUniAndBigram = uniAndBigramFeatureGen;
		DoccatModel cmUniAndBigram = Util.customTrainResource(TRAIN_FILE, 0, fgUniAndBigram);
		assertNotNull(cmUniAndBigram);
		DocumentCategorizerME catUniAndBigram = Util.customCategorizer(cmUniAndBigram, fgUniAndBigram);

		FeatureGenerator fgBigram = new BigramFeatureGenerator();
		DoccatModel cmBigram = Util
				.customTrainResource(TRAIN_FILE, 0, fgBigram);
		DocumentCategorizerME catBigram = Util.customCategorizer(
				cmUniAndBigram, fgBigram);

		FeatureGenerator[] fgUni = new FeatureGenerator[] { new BagOfWordsFeatureGenerator() };
		DoccatModel cmUni = Util.customTrainResource(TRAIN_FILE, 0, fgUni);
		DocumentCategorizerME catUni = Util.customCategorizer(cmUni, fgUni);

	
		testAllModels("crazy coffee", COFFEE, catUni, catBigram, catUniAndBigram);
		testAllModels("medium roast blends", COFFEE, catUni, catBigram, catUniAndBigram);
		testAllModels("coffee cans", COFFEE, catUni, catBigram, catUniAndBigram);
		
		testAllModels("juicy grapefruit", BEER, catUni, catBigram, catUniAndBigram);
		testAllModels("Yuengling Lager", BEER, catUni, catBigram, catUniAndBigram);
		
		// caramel shows up three times in BEER, and only once in COFFEE, preceeded by chocolate
		String caramelChoco = "caramel chocolate";
		
		// The unigram model (in our case) wrongly attributes this phrase to BEER
		assertEquals(BEER, Util.testAndGetBest(catUni, caramelChoco, fgUni));
		
		// The bigram feature "caramel chocolate" in the training set adds enough weight to correctly label it as COFFEE
		assertEquals(COFFEE, Util.testAndGetBest(catBigram, caramelChoco, fgBigram));
		assertEquals(COFFEE, Util.testAndGetBest(catUniAndBigram, caramelChoco, fgUniAndBigram));
		
	}
}
