#include <gtest/gtest.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/DataGenerator.h>

using namespace shogun;
using ::testing::Environment;

class CMockData
{
	public:
		CMockData() { };
		
		~CMockData() { };

		void generate_data(const index_t num_samples)
		{
			CMath::init_random(5);
			SGMatrix<float64_t> data =
				CDataGenerator::generate_gaussians(num_samples, 2, 2);
			CDenseFeatures<float64_t> features(data);

			SGVector<index_t> train_idx(num_samples), test_idx(num_samples);
			SGVector<float64_t> labels(num_samples);
			
			for (index_t i = 0, j = 0; i < data.num_cols; ++i)
			{
				if (i % 2 == 0)
					train_idx[j] = i;
				else
					test_idx[j++] = i;

				labels[i/2] = (i < data.num_cols/2) ? 1.0 : -1.0;
			}

			features_train = (CDenseFeatures<float64_t>*)features.copy_subset(train_idx);
			features_test = (CDenseFeatures<float64_t>*)features.copy_subset(test_idx);

			CBinaryLabels temp_labels = CBinaryLabels(labels);
			labels_train = (CBinaryLabels*)temp_labels.clone();
			labels_test = (CBinaryLabels*)temp_labels.clone();
		}

		/* get the traning features */
		CDenseFeatures<float64_t>* get_features_train()
		{
			return features_train;
		}

		/* get the test features */
		CDenseFeatures<float64_t>* get_features_test()
		{
			return features_test;
		}

		/* get the test labels */
		CBinaryLabels* get_labels_train()
		{
			return labels_train;
		}

		/* get the traning labels */
		CBinaryLabels* get_labels_test()
		{
			return labels_test;
		}

	protected:
		/* data for training */
		CDenseFeatures<float64_t>* features_train;

		/* data for testing */
		CDenseFeatures<float64_t>* features_test;

		/* traning label */
		CBinaryLabels* labels_train;

		/* testing label */
		CBinaryLabels* labels_test;
};

CMockData* mockData = nullptr;

class CBinaryLabelData : public Environment
{
	public:
		CBinaryLabelData() { };
		
		~CBinaryLabelData() { };

		// Override this to define how to set up the environment.
		virtual void SetUp() 
		{
			printf("Environment SetUp!\n");
			mockData = new CMockData();
			generate_data(100);
		}
		
		// Override this to define how to tear down the environment.
		virtual void TearDown() 
		{
			printf("Environment TearDown!\n");
		}

};