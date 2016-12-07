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

// includes
// std
#include <iostream>

// boost
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Eigen_Gurobi
#include <boost/test/unit_test.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/test/floating_point_comparison.hpp>

// Eigen
#include <Eigen/Dense>

// eigen-quadprog
#include <Gurobi.h>


struct QP1
{
	QP1()
	{
		nrvar = 6;
		nreq = 3;
		nrineq = 2;

		Q.resize(nrvar, nrvar);
		Aeq.resize(nreq, nrvar);
		Aineq.resize(nrineq, nrvar);

		C.resize(nrvar);
		Beq.resize(nreq);
		Bineq.resize(nrineq);
		XL.resize(nrvar);
		XU.resize(nrvar);
		X.resize(nrvar);


		Aeq << 1., -1., 1., 0., 3., 1.,
			-1., 0., -3., -4., 5., 6.,
			2., 5., 3., 0., 1., 0.;
		Beq << 1., 2., 3.;

		Aineq << 0., 1., 0., 1., 2., -1.,
			-1., 0., 2., 1., 1., 0.;
		Bineq << -1., 2.5;

		//with  x between ci and cs:
		XL << -1000., -10000., 0., -1000., -1000.,-1000.;
		XU << 10000., 100., 1.5, 100., 100., 1000.;

		//and minimize 0.5*x'*Q*x + p'*x with
		C << 1., 2., 3., 4., 5., 6.;
		Q.setIdentity();

		X << 1.7975426, -0.3381487, 0.1633880, -4.9884023, 0.6054943, -3.1155623;
	}

	int nrvar, nreq, nrineq;
	Eigen::MatrixXd Q, Aeq, Aineq;
	Eigen::VectorXd C, Beq, Bineq, XL, XU, X;
};


BOOST_AUTO_TEST_CASE(GurobiDense)
{
	QP1 qp1;

	int nrineq = static_cast<int>(qp1.Aineq.rows());

	std::cout << "Constructing" << std::endl;
	Eigen::GurobiDense qp(qp1.nrvar, qp1.nreq, nrineq);

	std::cout << "Solve" << std::endl;
	qp.solve(qp1.Q, qp1.C,
		qp1.Aeq, qp1.Beq,
		qp1.Aineq, qp1.Bineq,
		qp1.XL, qp1.XU);

	BOOST_CHECK_SMALL((qp.result() - qp1.X).norm(), 1e-6);
}

BOOST_AUTO_TEST_CASE(GurobiSparse)
{
	QP1 qp1;

	Eigen::GurobiSparse qp(qp1.nrvar, qp1.nreq, qp1.nrineq);

	Eigen::SparseMatrix<double> SQ(qp1.Q.sparseView());
	Eigen::SparseMatrix<double> SC(qp1.C.sparseView());
	Eigen::SparseMatrix<double> SAeq(qp1.Aeq.sparseView());
	Eigen::SparseMatrix<double> SAineq(qp1.Aineq.sparseView());
	Eigen::SparseVector<double> SBeq(qp1.Beq.sparseView());
	Eigen::SparseVector<double> SBineq(qp1.Bineq.sparseView());

	SQ.makeCompressed();
	SC.makeCompressed();
	SAeq.makeCompressed();
	SAineq.makeCompressed();

	qp.solve(SQ, SC,
		SAeq, SBeq,
		SAineq, SBineq,
		qp1.XL, qp1.XU);

	BOOST_CHECK_SMALL((qp.result() - qp1.X).norm(), 1e-6);
}

BOOST_AUTO_TEST_CASE(SolverParameters)
{
	QP1 qp1;

	int nrineq = static_cast<int>(qp1.Aineq.rows());
	double tol = 1e-8;
	int warmStart = 2;

	std::cout << "Constructing" << std::endl;
	Eigen::GurobiDense qp(qp1.nrvar, qp1.nreq, nrineq);
	qp.displayOutput(false);
	qp.warmStart(warmStart);
	qp.feasibilityTolerance(tol);
	qp.optimalityTolerance(tol);
	qp.inform();

	BOOST_CHECK_SMALL(tol - qp.feasibilityTolerance(), 1e-8);
	BOOST_CHECK_SMALL(tol - qp.optimalityTolerance(), 1e-8);
	BOOST_CHECK_EQUAL(warmStart, qp.warmStart());

	std::cout << "Solve" << std::endl;
	BOOST_REQUIRE(qp.solve(qp1.Q, qp1.C, qp1.Aeq, qp1.Beq, qp1.Aineq, qp1.Bineq, qp1.XL, qp1.XU));

	BOOST_CHECK_SMALL((qp.result() - qp1.X).norm(), 1e-6);
	qp.inform();
}