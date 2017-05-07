#include <shogun/classifier/svm/SVMOcas.h>
#include <shogun/features/DataGenerator.h>
#include <shogun/features/DenseFeatures.h>
#include <utils/binaryLabelData.h>
#include <gtest/gtest.h>

using namespace shogun;

#ifdef USE_GPL_SHOGUN
#ifdef HAVE_LAPACK

TEST(SVMOcasTest,train)
{
	index_t num_samples = 50;
	CMath::init_random(5);

	SGVector<index_t> train_idx(num_samples), test_idx(num_samples);
	SGVector<float64_t> labels(num_samples);

	CDenseFeatures<float64_t>* train_feats = wwdata->get_features_train();
	CDenseFeatures<float64_t>* test_feats = wwdata->get_features_test();
 
	CBinaryLabels* ground_truth = wwdata->get_labels_test();

	CSVMOcas* ocas = new CSVMOcas(1.0, train_feats, ground_truth);
	ocas->parallel->set_num_threads(1);
	ocas->set_epsilon(1e-5);
	ocas->train();
	float64_t objective = ocas->compute_primal_objective();

	EXPECT_NEAR(objective, 0.022321841487323236, 1e-2);

	CLabels* pred = ocas->apply(test_feats);
	for (int i = 0; i < num_samples; ++i)
		EXPECT_EQ(ground_truth->get_int_label(i), ((CBinaryLabels*)pred)->get_int_label(i));

	SG_UNREF(ocas);
	SG_UNREF(train_feats);
	SG_UNREF(test_feats);
	SG_UNREF(pred);
}
#endif // HAVE_LAPACK
#endif //USE_GPL_SHOGUN

