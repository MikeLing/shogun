#include "PermutationMatrix.h"


#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>

using namespace shogun;
using namespace Eigen;

bool is_permutation_matrix(SGMatrix<float64_t> m)
{
	Map<MatrixXd> mat(m.matrix,m.num_rows,m.num_cols);

	// scale
	for(int t = 0; t < mat.cols(); t++)
		mat.col(t) /= mat.col(t).cwiseAbs().maxCoeff();

	// round
	for (index_t i = 0; i < mat.rows(); i++)
	{
		for (index_t j = 0; j < mat.cols(); j++)
		{
			if (CMath::abs(CMath::round(mat(i,j))) >= 1.0)
				mat(i,j) = 1.0;
			else
				mat(i,j) = 0.0;
		}
	}

	// check only a single 1 per row
	for (index_t i = 0; i < mat.rows(); i++)
	{
		int num_ones = 0;
		for (index_t j = 0; j < mat.cols(); j++)
		{
			if (mat(i,j) >= 1.0)
				num_ones++;
		}

		if (num_ones != 1)
			return false;
	}

	// check only a single 1 per col
	for (index_t j = 0; j < mat.cols(); j++)
	{
		int num_ones = 0;
		for (index_t i = 0; i < mat.rows(); i++)
		{
			if (mat(i,j) >= 1.0)
				num_ones++;
		}

		if (num_ones != 1)
			return false;
	}

	return true;
}
