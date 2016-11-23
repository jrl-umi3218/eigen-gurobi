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

// associated header
#include "Gurobi.h"

namespace Eigen
{


/**
	*												GurobiCommon
	*/


GurobiCommon::GurobiCommon():
	Q_(),
	C_(),
	Beq_(),
	Bineq_(),
	X_(),
	fail_(0),
	nrvar_(0),
	nreq_(0),
	nrineq_(0),
	iter_(0),
	env_(),
	model_(env_)
{
}


int GurobiCommon::iter() const
{
	return iter_;
}


int GurobiCommon::fail() const
{
	return fail_;
}


const VectorXd& GurobiCommon::result() const
{
	return X_;
}


void GurobiCommon::problem(int nrvar, int nreq, int nrineq)
{
	for(int i = 0; i < nrvar_; ++i)
	{
	  model_.remove(*(vars_+i));
	}

	for(int i = 0; i < nreq_; ++i)
	{
	  model_.remove(*(eqconstr_+i));
	}

	for(int i = 0; i < nrineq_; ++i)
	{
	  model_.remove(*(ineqconstr_+i));
	}

	eqvars_.clear();
	ineqvars_.clear();
	lvars_.clear();
	rvars_.clear();

	nrvar_ = nrvar;
	nreq_ = nreq;
	nrineq_ = nrineq;

	Q_.resize(nrvar, nrvar);

	C_.resize(nrvar);
	Beq_.resize(nreq);
	Bineq_.resize(nrineq);
	X_.resize(nrvar);

	vars_ = model_.addVars(nrvar, GRB_CONTINUOUS);

	for(int i=0; i < Q_.size(); ++i)
	{
		lvars_.push_back(*(vars_+i/nrvar));
		rvars_.push_back(*(vars_+i%nrvar));
	}

	eqconstr_ = model_.addConstrs(nreq);
	std::vector<char> eqsense(static_cast<size_t>(nreq), '=');
	model_.set(GRB_CharAttr_Sense, eqconstr_, eqsense.data(), nreq);

	eqvars_.reserve(static_cast<size_t>(nrvar*nreq));
	for(int i = 0; i < nrvar; ++i) {
		for(int j = 0; j < nreq; ++j)
		{
			eqvars_.push_back(*(vars_+i));
		}
	}

	ineqconstr_ = model_.addConstrs(nrineq);
	std::vector<char> ineqsense(static_cast<size_t>(nrineq), '<');
	model_.set(GRB_CharAttr_Sense, ineqconstr_, ineqsense.data(), nrineq);

	for(int i = 0; i < nrvar; ++i) {
		for(int j = 0; j < nrineq; ++j)
		{
			ineqvars_.push_back(*(vars_+i));
		}
	}
}

/**
	*												GurobiDense
	*/


GurobiDense::GurobiDense()
{ }


GurobiDense::GurobiDense(int nrvar, int nreq, int nrineq)
{
	problem(nrvar, nreq, nrineq);
}


void GurobiDense::problem(int nrvar, int nreq, int nrineq)
{
	GurobiCommon::problem(nrvar, nreq, nrineq);
}


bool GurobiDense::solve(const MatrixXd& Q, const VectorXd& C,
	const MatrixXd& Aeq, const VectorXd& Beq,
	const MatrixXd& Aineq, const VectorXd& Bineq,
	const VectorXd& XL, const VectorXd& XU)
{
	//Objective
	GRBQuadExpr qexpr;
	qexpr.addTerms(Q.data(), rvars_.data(), lvars_.data(), static_cast<int>(Q.size()));

	GRBLinExpr lexpr;
	lexpr.addTerms(C.data(), vars_, nrvar_);
	model_.setObjective(0.5*qexpr+lexpr);

	model_.set(GRB_DoubleAttr_LB, vars_, XL.data(), nrvar_);
	model_.set(GRB_DoubleAttr_UB, vars_, XU.data(), nrvar_);

	//Update eq and ineq, column by column
    if (nreq_ > 0)
	    for(int i = 0; i < nrvar_; ++i)
		    model_.chgCoeffs(eqconstr_, eqvars_.data()+nreq_*i, Aeq.col(i).data(), static_cast<int>(Aeq.rows()));

    if (nrineq_ > 0)
	    for(int i = 0; i < nrvar_; ++i)
		    model_.chgCoeffs(ineqconstr_, ineqvars_.data()+nrineq_*i, Aineq.col(i).data(), static_cast<int>(Aineq.rows()));

	for(int i = 0; i < nreq_; ++i)
	{
		(eqconstr_+i)->set(GRB_DoubleAttr_RHS, Beq(i));
	}

	for(int i = 0; i < nrineq_; ++i)
	{
		(ineqconstr_+i)->set(GRB_DoubleAttr_RHS, Bineq(i));
	}

	model_.optimize();

	fail_ = model_.get(GRB_IntAttr_Status);
	bool success = fail_ == GRB_OPTIMAL;
	iter_ = model_.get(GRB_IntAttr_BarIterCount);
	double* result = model_.get(GRB_DoubleAttr_X, vars_, nrvar_);
	X_ = Map<VectorXd>(result, nrvar_);

	return success;
}


/**
	*												GurobiSparse
	*/


GurobiSparse::GurobiSparse()
{ }


GurobiSparse::GurobiSparse(int nrvar, int nreq, int nrineq)
{
  problem(nrvar, nreq, nrineq);
}


void GurobiSparse::problem(int nrvar, int nreq, int nrineq)
{
      GurobiCommon::problem(nrvar, nreq, nrineq);
}

bool GurobiSparse::solve(const SparseMatrix<double>& Q, const SparseVector<double>& C,
	const SparseMatrix<double>& Aeq, const SparseVector<double>& Beq,
	const SparseMatrix<double>& Aineq, const SparseVector<double>& Bineq,
	const VectorXd& XL, const VectorXd& XU)
{


	//Objective: quadratic terms
	GRBQuadExpr qexpr;
	for(int k = 0; k<Q.outerSize(); ++k)
	{
		for (SparseMatrix<double>::InnerIterator it(Q,k); it; ++it)
		{
			qexpr.addTerm(0.5*it.value(), *(vars_+it.row()), *(vars_+it.col()));
		}
	}

	//Objective: linear terms
	for (SparseVector<double>::InnerIterator it(C); it; ++it)
	{
		qexpr.addTerm(it.value(), *(vars_+it.row()));
	}

	model_.setObjective(qexpr);

	//Bounds
	model_.set(GRB_DoubleAttr_LB, vars_, XL.data(), nrvar_);
	model_.set(GRB_DoubleAttr_UB, vars_, XU.data(), nrvar_);

	//Update eq
	std::vector<double> zeros(static_cast<size_t>(nreq_), 0.0);
	for(int k = 0; k < Aeq.outerSize(); ++k)
	{
		std::cout << "Eq col : " << k << std::endl;
		model_.chgCoeffs(eqconstr_, eqvars_.data()+nreq_*k, zeros.data(), nreq_);
		for (SparseMatrix<double>::InnerIterator it(Aeq,k); it; ++it)
		{
			std::cout << "Changing : " << it.row() << ", " << it.col() << std::endl;
			model_.chgCoeff(*(eqconstr_+it.row()), *(vars_+it.col()), it.value());
		}
	}

	//Update ineq
	zeros.resize(static_cast<size_t>(nrineq_), 0.0);
	for(int k = 0; k < Aineq.outerSize(); ++k)
	{
		std::cout << "Ineq col : " << k << std::endl;
		model_.chgCoeffs(ineqconstr_, ineqvars_.data()+nrineq_*k, zeros.data(), nrineq_);
		for (SparseMatrix<double>::InnerIterator it(Aineq,k); it; ++it)
		{
			model_.chgCoeff(*(ineqconstr_+it.row()), *(vars_+it.col()), it.value());
		}
	}

	//Update RHSes
	for(SparseVector<double>::InnerIterator it(Beq); it; ++it)
	{
		(eqconstr_+it.row())->set(GRB_DoubleAttr_RHS, it.value());
	}

	for(SparseVector<double>::InnerIterator it(Bineq); it; ++it)
	{
		(ineqconstr_+it.row())->set(GRB_DoubleAttr_RHS, it.value());
	}

	model_.optimize();

	fail_ = model_.get(GRB_IntAttr_Status);
	bool success = fail_ == GRB_OPTIMAL;
	double* result = model_.get(GRB_DoubleAttr_X, vars_, nrvar_);
	iter_ = model_.get(GRB_IntAttr_BarIterCount);
	X_ = Map<VectorXd>(result, nrvar_);

	return success;
}

} // namespace Eigen
