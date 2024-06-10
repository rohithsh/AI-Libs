package ai.libs.jaicore.experiments.mlexample;

import java.io.File;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;


import org.aeonbits.owner.ConfigCache;
import org.aeonbits.owner.ConfigFactory;
import org.api4.java.algorithm.exceptions.AlgorithmExecutionCanceledException;
import org.api4.java.algorithm.exceptions.AlgorithmTimeoutedException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.libs.jaicore.db.IDatabaseConfig;
import ai.libs.jaicore.experiments.ExperimentDBEntry;
import ai.libs.jaicore.experiments.ExperimentDatabasePreparer;
import ai.libs.jaicore.experiments.ExperimentRunner;
import ai.libs.jaicore.experiments.IExperimentDatabaseHandle;
import ai.libs.jaicore.experiments.IExperimentIntermediateResultProcessor;
import ai.libs.jaicore.experiments.IExperimentSetEvaluator;
import ai.libs.jaicore.experiments.databasehandle.ExperimenterMySQLHandle;
import ai.libs.jaicore.experiments.exceptions.ExperimentAlreadyExistsInDatabaseException;
import ai.libs.jaicore.experiments.exceptions.ExperimentDBInteractionFailedException;
import ai.libs.jaicore.experiments.exceptions.IllegalExperimentSetupException;
import ai.libs.jaicore.logging.LoggerUtil;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.classifiers.*;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

public class MachineLearningExperimenter {

	/**
	 * Variables for the experiment and database setup
	 */
	private static final File configFile = new File("/Users/rohith/Desktop/IEM/AILibs/JAICore/jaicore-experiments/testrsc/mlexample/setup.properties");
	private static final IExampleMCCConfig m = (IExampleMCCConfig)ConfigCache.getOrCreate(IExampleMCCConfig.class).loadPropertiesFromFile(configFile);
	private static final IDatabaseConfig dbconfig = (IDatabaseConfig)ConfigFactory.create(IDatabaseConfig.class).loadPropertiesFromFile(configFile);
	private static final IExperimentDatabaseHandle dbHandle = new ExperimenterMySQLHandle(dbconfig);
	private static final Logger logger = LoggerFactory.getLogger(MachineLearningExperimenter.class);
	static Map<String, String> datasetMapping = new HashMap<>();

	public static void main(final String[] args) throws ExperimentDBInteractionFailedException, AlgorithmTimeoutedException, IllegalExperimentSetupException, ExperimentAlreadyExistsInDatabaseException, InterruptedException, AlgorithmExecutionCanceledException {
		mapDatasets();
		createTableWithExperiments();
		runExperiments();
	}

	private static void mapDatasets() {
		datasetMapping.put("SWAN_DATASET_NEW", "path_to_arff_file");
	}

	public static void createTableWithExperiments() throws ExperimentDBInteractionFailedException, AlgorithmTimeoutedException, IllegalExperimentSetupException, ExperimentAlreadyExistsInDatabaseException, InterruptedException, AlgorithmExecutionCanceledException {
		ExperimentDatabasePreparer preparer = new ExperimentDatabasePreparer(m, dbHandle);
		preparer.setLoggerName(LoggerUtil.LOGGER_NAME_EXAMPLE);
		preparer.synchronizeExperiments();
	}

	public static void deleteTable() throws ExperimentDBInteractionFailedException {
		dbHandle.deleteDatabase();
	}

	public static void runExperiments() throws ExperimentDBInteractionFailedException, InterruptedException {
		Random r = new Random(System.currentTimeMillis());
		ExperimentRunner runner = new ExperimentRunner(m, new IExperimentSetEvaluator() {

			@Override
			public void evaluate(final ExperimentDBEntry experimentEntry, final IExperimentIntermediateResultProcessor processor) throws InterruptedException, ClassNotFoundException, IllegalAccessException, InstantiationException {

                /* get experiment setup */
                Map<String, String> description = experimentEntry.getExperiment().getValuesOfKeyFields();
                String classifierName = description.get("classifiers");
                String datasetName = description.get("datasets");
                int seed = Integer.parseInt(description.get("seeds"));

                /* create objects for experiment */
                logger.info("Evaluate {} for dataset {} and seed {}", classifierName, datasetName, seed);

				/* Load the dataset */
				ConverterUtils.DataSource source;
				Instances data = null;
				switch (datasetName){
					case "SWAN_MEKA_CODE":
                        try {
							source = new ConverterUtils.DataSource("/Users/rohith/Desktop/IEM/meka-code.arff");
							data = source.getDataSet();
                        } catch (Exception e) {
                            throw new RuntimeException(e);
                        }
						break;
					default:
						logger.info("Invalid Dataset");
                }

				/* Set class indices */
//				for(int i=0; i<=10; i++){
//					data.setClassIndex(i);
//				}
				data.setClassIndex(5);

				// Train-test split (70:30)
				int trainSize = (int) Math.round(data.size() * 0.7);
				int testSize = data.size() - trainSize;
				Instances trainData = new Instances(data, 0, trainSize);
				Instances testData = new Instances(data, trainSize, testSize);

                /* create the classifier */
                Map<String, Object> results = new HashMap<>();
                long timeStartTraining = System.currentTimeMillis();
                Classifier classifier = (Classifier) Class.forName(classifierName).newInstance();
				try{
					classifier.buildClassifier(trainData);
					Evaluation evaluation = new Evaluation(testData);
					evaluation.evaluateModel(classifier, testData);
					results.put("accuracy", evaluation.pctCorrect()/100);
				} catch (Exception e){
					System.out.println(e);
				}
                results.put("traintime", System.currentTimeMillis() - timeStartTraining);
			}
		}, dbHandle);
		runner.randomlyConductExperiments(10);
	}
}
