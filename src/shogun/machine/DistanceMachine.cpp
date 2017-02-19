/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Christian Gehl
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST
 */

#include <shogun/machine/DistanceMachine.h>
#include <shogun/distance/Distance.h>
#include <shogun/base/Parameter.h>

using namespace shogun;

#ifdef HAVE_OPENMP
#include <omp.h>
#endif

CDistanceMachine::CDistanceMachine()
: CMachine()
{
	init();
}

CDistanceMachine::~CDistanceMachine()
{
	SG_UNREF(distance);
}

void CDistanceMachine::init()
{
	/* all distance machines should store their models, i.e. cluster centers
	 * At least, it has to be ensured, that after calling train(), or in the
	 * call of apply() in the cases where there is no train method, the lhs
	 * of the underlying distance is set to cluster centers */
	set_store_model_features(true);

	distance=NULL;
	m_parameters->add((CSGObject**)&distance, "distance", "Distance to use");
}

void CDistanceMachine::distances_lhs(float64_t* result,int32_t idx_a1,int32_t idx_a2,int32_t idx_b)
{
	int32_t num_vec=idx_a2-idx_a1+1;
	int32_t num_threads;
	int64_t step;
	#pragma omp parallel shared(num_threads, step)
	{
#ifdef HAVE_OPENMP
		#pragma opm single
		{
			num_threads=omp_get_num_threads();
			step=num_vec/num_threads;
			num_threads--;
		}
		int32_t thread_num=omp_get_thread_num();
#else
		num_threads=0;
		step=num_vec;
#endif
		int32_t start = thread_num*step+idx_a1;
		int32_t end = (thread_num==num_threads) ? idx_a2 : (thread_num+1)*step+idx_a1;
		for (int32_t i=start; i<end; i++)
			result[i] =distance->distance(i,idx_b);
	}
}

void CDistanceMachine::distances_rhs(float64_t* result,int32_t idx_b1,int32_t idx_b2,int32_t idx_a)
{
	int32_t num_vec=idx_b2-idx_b1+1;
	int32_t num_threads;
	int64_t step;
	#pragma omp parallel shared(num_threads, step)
	{
#ifdef HAVE_OPENMP
		#pragma opm single
		{
			num_threads=omp_get_num_threads();
			step=num_vec/num_threads;
			num_threads--;
		}
		int32_t thread_num=omp_get_thread_num();
#else
		num_threads=0;
		step=num_vec;
#endif
		int32_t start = thread_num*step+idx_b1;
		int32_t end = (thread_num==num_threads) ? idx_b2 : (thread_num+1)*step+idx_b1;
		for (int32_t i=start; i<end; i++)
			result[i] =distance->distance(i,idx_a);
	}
}

CMulticlassLabels* CDistanceMachine::apply_multiclass(CFeatures* data)
{
	if (data)
	{
		/* set distance features to given ones and apply to all */
		CFeatures* lhs=distance->get_lhs();
		distance->init(lhs, data);
		SG_UNREF(lhs);

		/* build result labels and classify all elements of procedure */
		CMulticlassLabels* result=new CMulticlassLabels(data->get_num_vectors());
		for (index_t i=0; i<data->get_num_vectors(); ++i)
			result->set_label(i, apply_one(i));
		return result;
	}
	else
	{
		/* call apply on complete right hand side */
		CFeatures* all=distance->get_rhs();
		CMulticlassLabels* result = apply_multiclass(all);
		SG_UNREF(all);
		return result;
	}
	return NULL;
}

float64_t CDistanceMachine::apply_one(int32_t num)
{
	/* number of clusters */
	CFeatures* lhs=distance->get_lhs();
	int32_t num_clusters=lhs->get_num_vectors();
	SG_UNREF(lhs);

	/* (multiple threads) calculate distances to all cluster centers */
	float64_t* dists=SG_MALLOC(float64_t, num_clusters);
	distances_lhs(dists, 0, num_clusters-1, num);

	/* find cluster index with smallest distance */
	float64_t result=dists[0];
	index_t best_index=0;
	for (index_t i=1; i<num_clusters; ++i)
	{
		if (dists[i]<result)
		{
			result=dists[i];
			best_index=i;
		}
	}

	SG_FREE(dists);

	/* implicit cast */
	return best_index;
}

void CDistanceMachine::set_distance(CDistance* d)
{
	SG_REF(d);
	SG_UNREF(distance);
	distance=d;
}

CDistance* CDistanceMachine::get_distance() const
{
	SG_REF(distance);
	return distance;
}

void CDistanceMachine::store_model_features()
{
	SG_ERROR("store_model_features not yet implemented for %s!\n",
	         get_name());
}

