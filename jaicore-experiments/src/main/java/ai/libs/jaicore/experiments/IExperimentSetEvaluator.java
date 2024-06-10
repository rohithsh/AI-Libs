package ai.libs.jaicore.experiments;

import ai.libs.jaicore.experiments.exceptions.ExperimentEvaluationFailedException;
import ai.libs.jaicore.experiments.exceptions.ExperimentFailurePredictionException;

public interface IExperimentSetEvaluator {

	/**
	 * Method to compute a single point of the experiment set
	 *
	 * @param experimentEntry The point of the experiment set
	 * @param processor A handle to return intermediate results to the experiment runner routine
	 * @throws Exception
	 */
	public void evaluate(ExperimentDBEntry experimentEntry, IExperimentIntermediateResultProcessor processor) throws ExperimentEvaluationFailedException, ExperimentFailurePredictionException, InterruptedException, ClassNotFoundException, IllegalAccessException, InstantiationException;
}
