// This file is part of EigenQP.
//
// EigenQP is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// EigenQP is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with EigenQP.  If not, see <http://www.gnu.org/licenses/>.

#pragma once

// includes
// Eigen
#include <Eigen/Core>
#include <Eigen/Sparse>

//Gurobi
#include <gurobi_c++.h>

// eigen-quadprog
#include "eigen_gurobi_api.h"

namespace Eigen
{

class GurobiCommon
{
public:
	EIGEN_GUROBI_API GurobiCommon();

	EIGEN_GUROBI_API int iter() const;
	EIGEN_GUROBI_API int fail() const;

	EIGEN_GUROBI_API const VectorXd& result() const;

	EIGEN_GUROBI_API void problem(int nrvar, int nreq, int nrineq);

protected:
	void fillQCBf(int nreq, int nrineq,
		const MatrixXd& Q, const VectorXd& C,
		const VectorXd& Beq, const VectorXd& Bineq);

protected:
	MatrixXd Q_;
	VectorXd C_, Beq_, Bineq_, X_;
	int fail_, nrvar_, nreq_, nrineq_, iter_;

	GRBEnv env_;
	GRBModel model_;

	GRBVar* vars_;
	std::vector<GRBVar> lvars_, rvars_, eqvars_, ineqvars_;
	GRBConstr* eqconstr_;
	GRBConstr* ineqconstr_;
};


class GurobiDense : public GurobiCommon
{
public:
	EIGEN_GUROBI_API GurobiDense();
	EIGEN_GUROBI_API GurobiDense(int nrvar, int nreq, int nrineq);

	EIGEN_GUROBI_API void problem(int nrvar, int nreq, int nrineq);

	EIGEN_GUROBI_API bool solve(const MatrixXd& Q, const VectorXd& C,
		const MatrixXd& Aeq, const VectorXd& Beq,
		const MatrixXd& Aineq, const VectorXd& Bineq,
		const VectorXd& XL, const VectorXd& XU);
};


class GurobiSparse : public GurobiCommon
{
public:
	EIGEN_GUROBI_API GurobiSparse();
	EIGEN_GUROBI_API GurobiSparse(int nrvar, int nreq, int nrineq);

	EIGEN_GUROBI_API void problem(int nrvar, int nreq, int nrineq);

	EIGEN_GUROBI_API bool solve(const SparseMatrix<double>& Q, const SparseVector<double>& C,
		const SparseMatrix<double>& Aeq, const SparseVector<double>& Beq,
		const SparseMatrix<double>& Aineq, const SparseVector<double>& Bineq,
		const VectorXd& XL, const VectorXd& XU);
};

} // namespace Eigen
